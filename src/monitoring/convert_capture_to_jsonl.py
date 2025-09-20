# convert_capture_to_jsonl.py
import gzip
import json


def parse_payload(s):
    # s 是 endpointInput.data（字符串）
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        # 不是 JSON，就当成 CSV 或纯文本，交给后续处理（此处直接返回 None）
        return None

    # 1) sklearn dataframe_split 格式
    try:
        dfs = obj["inputs"]["dataframe_split"]
        cols = dfs["columns"]
        rows = dfs["data"]
        for row in rows:
            yield dict(zip(cols, row))
        return
    except Exception:
        pass

    # 2) {"instances": [...]}：list[dict] 或 list[list]
    if isinstance(obj, dict) and "instances" in obj:
        inst = obj["instances"]
        if inst and isinstance(inst[0], dict):
            for r in inst:
                yield r
            return
        if inst and isinstance(inst[0], list):
            for r in inst:
                yield {f"f{i}": v for i, v in enumerate(r)}
            return

    # 3) 直接是扁平 dict
    if isinstance(obj, dict):
        yield obj
        return

    # 其他情况：放弃
    return


def iter_capture_records(fp):
    # DataCapture 既可能是 NDJSON，也可能一整个 JSON 对象
    raw = fp.read()
    text = raw.decode("utf-8", errors="replace")
    # 粗暴判断是否是单个 JSON 对象
    text_strip = text.strip()
    if text_strip.startswith("{") and text_strip.endswith("}"):
        rec = json.loads(text_strip)
        yield rec
        return
    # 否则按行切（NDJSON）
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def convert(in_path, out_path):
    # 支持本地文件（.jsonl/.json/.gz）路径
    opener = open
    if in_path.endswith(".gz"):
        opener = gzip.open
    with opener(in_path, "rb") as fp, open(out_path, "w", encoding="utf-8") as out:
        n_in, n_out = 0, 0
        for rec in iter_capture_records(fp):
            n_in += 1
            cd = rec.get("captureData", {})
            ein = cd.get("endpointInput", {})
            data = ein.get("data")
            if not isinstance(data, str):
                continue
            for flat in parse_payload(data) or []:
                out.write(json.dumps(flat, ensure_ascii=False) + "\n")
                n_out += 1
        print(f"converted {n_in} capture records -> {n_out} jsonl rows at {out_path}")


if __name__ == "__main__":
    # 改成你的本地文件名
    in_path = r"./48-14-999-99f083c7-0449-4211-bd52-5a6c46fb80e7.jsonl"
    out_path = r"./baseline_flat.jsonl"
    convert(in_path, out_path)
