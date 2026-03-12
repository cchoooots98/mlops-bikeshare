# dbt 生产差距整改与 AWS 迁移备注

## 1. 当前已经达到的状态

当前仓库中的 dbt 相关链路已经具备以下能力：

- 高频 `dbt_feature_build_5min` 已经与低频 `dbt_quality_hourly` 分离
- 高频链路已经具备 weather semantic freshness gate
- dbt tests 已经按 `hf_smoke` 与 `quality_gate` 分层
- 高风险 anomaly mart 已从高频 build 中剥离
- duplicate / capacity mismatch 已引入 recent-window rate threshold tests
- Airflow 中已经增加串行 pool，避免高频与低频 dbt 任务并发冲突

这些能力说明项目已经脱离“纯手动开发态”，进入了可持续运行的生产化初级阶段。

---

## 2. 现在仍然不符合企业生产级标准的点

### 2.1 缺少正式告警链路

当前 DAG 失败、freshness 超阈值、quality gate 失败时，主要依赖 Airflow task failure 和日志输出。

问题：

- 系统知道失败，但人不一定知道
- 不能保证在非工作时间及时响应
- 无法将告警分级为 warning / critical

企业级解决方案：

- 为 `dbt_feature_build_5min` 和 `dbt_quality_hourly` 增加 Airflow failure callback
- 在本地阶段至少接 Email / Slack webhook
- 迁移 AWS 后接 CloudWatch Alarm + SNS

推荐优先级：高

---

### 2.2 缺少 outage/backfill 恢复机制

当前有正常调度链路，但没有专门的“缺失窗口识别 + 补跑”机制。

问题：

- 网络中断几小时后，只知道数据断了，但没有标准恢复路径
- 人工判断补哪些窗口容易出错
- 一旦未来迁移 AWS，没有恢复 runbook 会让值班和运营成本变高

企业级解决方案：

- 增加一个 `data_gap_assessment` DAG 或脚本
- 检测 `station_status` / `weather_current` / `weather_hourly` 最近窗口缺失情况
- 输出缺失时间段清单
- 如果 raw 历史可恢复，则只补缺失窗口
- 如果 raw 历史不可恢复，则记录 gap window，并在训练时排除

推荐优先级：高

---

### 2.3 缺少训练与晋升的质量联动

当前 quality DAG 失败后，会影响后续人工判断，但还没有硬性接入：

- 离线训练是否允许继续
- candidate 模型是否允许注册 / 晋升

企业级解决方案：

- 在离线重训练入口前增加质量状态检查
- 最近一轮 `dbt_quality_hourly` 失败时，禁止训练继续
- promotion 到 staging / prod 前，必须检查最近质量 DAG 成功

推荐优先级：高

---

### 2.4 高频 DAG 仍然依赖全 parents build

当前 `hf_feature_build_models` 会带上 `feat_station_snapshot_*` 的全部 parents。

问题：

- 当前数据量还能接受
- 但未来数据量增长后，5 分钟 SLA 可能再次失守

企业级解决方案：

- 后续将高频 DAG 进一步拆成：
  - 高频只刷新 feature 所需增量模型
  - 低频刷新较稳定的 dim / supporting marts
- 对最重模型增加基准测试和耗时阈值

推荐优先级：中

---

### 2.5 缺少系统级质量与性能指标落地

当前已经能在日志里看到：

- `WEATHER_FRESHNESS_SUMMARY`
- `DBT_FEATURE_BUILD_SUMMARY`
- `DBT_QUALITY_GATE_SUMMARY`

问题：

- 日志不是指标
- 无法长期画趋势图
- 无法自动做阈值告警

企业级解决方案：

- 将这些摘要转成可观测指标
- 在 AWS 上接到 CloudWatch Metrics
- 至少记录：
  - feature build duration
  - smoke test duration
  - quality DAG duration
  - weather current age
  - weather hourly age
  - forecast coverage minutes
  - duplicate max rate
  - capacity mismatch max rate

推荐优先级：中

---

### 2.6 缺少数据保留与重建策略文档

当前对于以下问题还没有形成统一规则：

- raw 表保留多久
- analytics schema 出问题时是全重建还是局部重建
- outage 后是补 raw 还是只重跑 dbt

企业级解决方案：

- 增加数据保留和恢复 runbook
- 定义：
  - raw/staging/analytics 的保留周期
  - 允许全量清库的场景
  - 不允许清库、必须局部 backfill 的场景

推荐优先级：中

---

## 3. 对“几小时中断后是否要清空所有数据重来”的标准结论

默认结论：**不要清空所有数据。**

原因：

- `station_status` 和 `weather` 都是时间序列流数据
- 如果缺的是历史窗口，清空当前库并不能自动恢复这些历史
- 反而会把已经正确落地的数据也一起删除

企业级标准做法：

1. 先保留现有 raw / staging / analytics
2. 识别中断窗口
3. 判断 raw 历史是否可恢复
4. 可恢复则只补缺失窗口
5. 不可恢复则保留现状，并在训练中排除 gap window

只有以下情况才建议全清：

- staging 或 analytics 结构已经错乱
- 原始去重逻辑长期错误，历史整体不可信
- 你明确在做全量重建演练

---

## 4. 迁移 AWS 时建议的目标形态

### 4.1 运行面

- Airflow 迁移到 MWAA，或保留自管 Airflow on EC2/ECS
- Postgres 迁移到 RDS PostgreSQL
- raw 文件和归档统一落到 S3

### 4.2 监控面

- Airflow logs 进入 CloudWatch Logs
- DAG failure 接 SNS / Email / Slack
- 自定义指标进入 CloudWatch Metrics
- 对以下指标建 Alarm：
  - 高频 DAG 失败
  - 低频质量 DAG 失败
  - weather semantic freshness 超 error
  - duplicate rate 超 error
  - capacity mismatch rate 超 error
  - feature build duration 超 SLA

### 4.3 安全面

- dbt / Airflow / MLflow / database 凭据进 Secrets Manager 或 SSM Parameter Store
- 不再依赖本地 `.env`

### 4.4 恢复面

- S3 保留 raw archive
- 有专门的 gap assessment / backfill DAG
- 训练链路依赖最近质量 DAG 成功

---

## 5. 推荐的后续执行顺序

1. 增加 gap assessment / backfill runbook 或 DAG
2. 增加 Airflow failure callback 与正式告警
3. 将质量状态接入离线训练与模型晋升 gate
4. 将关键摘要日志转成指标
5. 规划迁移 AWS 后的 CloudWatch / SNS / Secrets Manager / RDS 方案

---

## 6. 一句话结论

当前 dbt 与 dbt DAG 已经具备企业生产化的核心结构，但还缺：

- 正式告警
- 恢复机制
- 训练联动 gate
- 指标化监控

补完这 4 项，才算真正接近企业级稳定运行标准。
