# Demo Video Script — Bikeshare Station Risk Monitor

**Format**: 3-minute Loom screen recording with voiceover
**Target audience**: Hiring managers, technical reviewers, portfolio visitors
**Tone**: Concise, confident, technical but accessible

---

## Pre-Recording Setup Checklist

- [ ] Open the dashboard in Chrome at full width (1920x1080 recommended)
- [ ] Confirm the environment badge shows **PRODUCTION** (not STAGING)
- [ ] Select **Bike stockout** as the prediction target in the sidebar
- [ ] Set Top-N stations slider to **20**
- [ ] Set History snapshots slider to **24**
- [ ] Navigate to the **Live Ops** tab so the map is visible on screen
- [ ] Close all other browser tabs and mute desktop notifications
- [ ] Test microphone levels — speak at normal conversational volume
- [ ] Have the map zoomed to show all of Paris with station markers visible

---

## Script

### Scene 1 — Opening [0:00 – 0:25]

**Screen**: Live Ops tab with the full Paris station map visible. Status cards at the top showing endpoint, model version, and latest prediction time.

> *"This is a real-time MLOps platform I built that predicts bikeshare station stockouts 30 minutes into the future — for over 1,400 Paris Velib stations."*

**Action**: Slowly move the cursor across the top status cards (Pipeline state → Prediction target → Active endpoint → Active model → Latest prediction).

> *"The system runs end-to-end: live data ingestion every 5 minutes from the Velib GBFS feed and OpenWeather API, dbt-powered feature engineering in PostgreSQL, XGBoost models served through SageMaker, and continuous quality monitoring — all orchestrated by Airflow on EC2."*

**Action**: As you say "orchestrated by Airflow", gesture briefly toward the "Healthy" pipeline state badge.

> *"Let me walk you through the live dashboard."*

---

### Scene 2 — Live Ops Map [0:25 – 0:55]

**Screen**: The Folium map with colored station markers.

> *"Each dot on this map is a Velib station. The color represents its current stockout risk: red means high probability of running out of bikes in the next 30 minutes, orange is elevated risk, and teal is normal."*

**Action**: Hover over a **red** station marker. Wait for the tooltip popup to appear (shows station name, ID, current bikes, capacity, status, risk level, probability score).

> *"When I hover over a station, I can see its current inventory, capacity, and the model's predicted stockout probability. This station has [read the probability] — it is flagged as critical."*

**Action**: Move to hover over a **teal** station for contrast.

> *"Compare that to this station — plenty of bikes, low risk score. The predictions refresh every 15 minutes from the production SageMaker endpoint."*

**Action**: Scroll down slightly to reveal the alert banner and the Top-N risk table.

> *"Below the map, the top-risk table ranks the stations most likely to experience a stockout right now. An operations team would use this to prioritize where to send rebalancing trucks."*

---

### Scene 3 — Station Deep Dive [0:55 – 1:25]

**Action**: Click on a station marker (preferably one with some variation in its recent history — not flat at zero). The Station History section should update.

> *"If I click on a specific station..."*

**Action**: Click the **Station History** tab at the top.

> *"...I get its prediction history over time. This chart shows the model's confidence score for the last 24 snapshots."*

**Screen**: Station History chart with the time-series line and the dashed orange threshold line.

> *"The dashed orange line is the model-specific alert threshold — learned during training to optimize recall. Anything above this line would trigger an alert."*

**Action**: Point the cursor along the time-series line, following its trajectory.

> *"You can see the score fluctuating with demand patterns — rising during commute hours and settling down at quieter times. This time-series view helps operators understand not just the current risk, but the trend."*

---

### Scene 4 — Prediction Quality [1:25 – 1:55]

**Action**: Click the **Prediction Quality** tab.

**Screen**: The quality status panel (green banner if quality data is available) and the metric cards below it.

> *"The Prediction Quality tab shows how the model is actually performing against ground truth. The system runs a quality backfill every 15 minutes — once the real outcome is known after the 30-minute horizon, it scores the predictions."*

**Action**: Point to the PR-AUC card, then the F1 card.

> *"PR-AUC is our primary metric — currently at [read the value]. F1 at threshold is [read the value]. Both are within healthy bounds."*

**Action**: Point to the Threshold Hit Rate and Samples cards.

> *"Threshold Hit Rate shows what fraction of predictions exceed the alert boundary — about [read the value], which tells us roughly how often the model flags a station. And the Samples count confirms we are scoring all [read value] station-windows per day."*

**Action**: Briefly click the sidebar and switch the prediction target to **Dock stockout**, then switch back to **Bike stockout**.

> *"The platform supports dual targets — bikes and docks each have independent models, endpoints, and quality metrics. I can switch between them in the sidebar."*

---

### Scene 5 — System Health [1:55 – 2:25]

**Action**: Click the **System Health** tab.

**Screen**: The metric grid showing ModelLatency, Invocation errors, Invocations, Prediction Heartbeat, and PSI metrics.

> *"System Health pulls CloudWatch metrics directly. The top row shows SageMaker endpoint performance."*

**Action**: Point to the ModelLatency card.

> *"Model latency p95 is [read value] milliseconds — well under our 200-millisecond SLA. Zero 5xx errors, zero 4xx errors."*

**Action**: Point to the Invocations card.

> *"The Invocations count confirms the endpoint is receiving traffic from the Airflow serving DAG every 15 minutes."*

**Action**: Point to the PSI cards (PSI Overall, PSI Core, PSI Weather).

> *"The bottom row tracks feature drift using Population Stability Index, split into core features and weather features. PSI Core at [read value] is well below the 0.20 warning threshold. Weather PSI is advisory — it naturally fluctuates with the seasons but does not trigger alerts on its own."*

**Action**: Point to the Prediction Heartbeat card.

> *"Behind the scenes, CloudWatch alarms monitor all of these metrics. If latency spikes, errors appear, or quality degrades, SNS sends an alert before the operator notices."*

---

### Scene 6 — Data Pipeline Status [2:25 – 2:50]

**Action**: Click the **Data Status** tab.

**Screen**: The freshness table showing Prediction artifact, Quality artifact, and feat_station_snapshot_latest.

> *"The Data Status tab monitors the full data pipeline end-to-end."*

**Action**: Point to each row in the table.

> *"Each row shows a data source, its last update time, the delay in minutes, and whether it meets the operator SLA. The prediction artifact refreshes every 15 minutes and is stale after 30. Quality evidence has a built-in 30-minute label maturity window plus a 7-minute backfill lag. The feature table tracks serving feature freshness."*

**Action**: Point to the Status column.

> *"This closes the observability loop: data freshness feeds model freshness feeds prediction freshness. Any break in the chain — a failed GBFS fetch, a stuck dbt build, a SageMaker timeout — surfaces here immediately."*

---

### Scene 7 — Closing [2:50 – 3:00]

**Action**: Click back to the **Live Ops** tab for a clean closing frame. The full Paris map with colored stations is visible.

> *"That is the system: live ingestion, automated feature engineering, ML serving with continuous monitoring, and a single dashboard that ties operations, model quality, and data health together."*

> *"The full architecture, model cards, deployment guide, and operations runbook are all in the repository. Thanks for watching."*

**Action**: Hold on the Live Ops map for 2–3 seconds as a clean closing frame.

---

## Post-Recording Checklist

- [ ] Trim dead air at the start and end of the recording
- [ ] Verify audio is clear and consistent throughout
- [ ] Check that all tooltips and tab transitions rendered cleanly
- [ ] Add the Loom link to `DEMO.md` (replace the `(#)` placeholder in the Demo Video section)
- [ ] Optionally add the Loom link to the GitHub repo "About" section
