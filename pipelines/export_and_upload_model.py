# pipelines/export_and_upload_model.py
# Purpose:
#   Export a model from a local MLflow SQLite DB *without importing mlflow*.
#   Works even if model_versions.source is "models:/..." (registry ref).
#   It falls back to the run's artifact_uri and auto-detects the model dir
#   by scanning for a child folder that contains an "MLmodel" file.
#
# Steps:
#   1) Query model_versions for latest version in a given stage (default: Staging).
#   2) If source starts with file://, use it.
#      Else:
#        - read run_id from the same row,
#        - read runs.artifact_uri for that run,
#        - scan subfolders for a directory that contains "MLmodel".
#   3) Tar that directory into model.tar.gz and upload to S3.
#
# Usage (PowerShell):
#   python pipelines/export_and_upload_model.py `
#     --mlflow-db "E:/算法自学/End2EndProject/mlops-bikeshare/mlflow.db" `
#     --model-name bikeshare_risk `
#     --s3-uri s3://mlflow-sagemaker-ca-central-1-387706002632/export/bikeshare_risk/model.tar.gz `
#     --region ca-central-1

import argparse
import os
import sqlite3
import tarfile
from urllib.parse import urlparse

import boto3


def file_uri_to_path(uri: str) -> str:
    """Convert file:///E:/path/to/dir to a Windows path E:\\path\\to\\dir."""
    p = urlparse(uri)
    if p.scheme != "file":
        raise ValueError(f"Not a file:// URI: {uri}")
    path = p.path
    # On Windows, urlparse gives "/E:/..." -> strip the first slash
    if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path

def find_model_dir_from_run_artifacts(artifacts_root: str) -> str:
    """
    Given a run's artifacts root (local path), find a child directory
    that looks like an MLflow model (contains an 'MLmodel' file).
    Return the first match found (depth=1 and depth=2 to cover common cases).
    """
    if not os.path.isdir(artifacts_root):
        raise FileNotFoundError(f"Artifacts root does not exist: {artifacts_root}")

    # Depth 1 search
    for name in os.listdir(artifacts_root):
        sub = os.path.join(artifacts_root, name)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "MLmodel")):
            return sub

    # Depth 2 search (sometimes models are in artifacts/model or artifacts/sklearn-model)
    for name in os.listdir(artifacts_root):
        sub = os.path.join(artifacts_root, name)
        if os.path.isdir(sub):
            for name2 in os.listdir(sub):
                sub2 = os.path.join(sub, name2)
                if os.path.isdir(sub2) and os.path.isfile(os.path.join(sub2, "MLmodel")):
                    return sub2

    raise FileNotFoundError(
        f"Could not find a folder containing 'MLmodel' under: {artifacts_root}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow-db", required=True, help="Path to mlflow SQLite DB file.")
    ap.add_argument("--model-name", required=True, help="Registered model name.")
    ap.add_argument("--stage", default="Staging", help="Stage to pick (default: Staging).")
    ap.add_argument("--s3-uri", required=True, help="Destination S3 URI for model.tar.gz.")
    ap.add_argument("--region", default="ca-central-1", help="AWS region.")
    ap.add_argument("--model-dir", help="Explicit MLflow model directory (with MLmodel file)")
    args = ap.parse_args()

    # 1) Look up latest version in desired stage
    conn = sqlite3.connect(args.mlflow_db)
    cur = conn.cursor()
    cur.execute("""
        SELECT version, source, run_id
        FROM model_versions
        WHERE name = ? AND current_stage = ?
        ORDER BY CAST(version AS INTEGER) DESC
        LIMIT 1
    """, (args.model_name, args.stage))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise SystemExit(f"No version found for model '{args.model_name}' in stage '{args.stage}'.")
    version, source, run_id = row
    print(f"Found {args.model_name} v{version} in stage {args.stage}")
    print(f"source: {source}")
    print(f"run_id: {run_id}")

    model_dir = None

    # 2) If source is a file:// URI, use it directly
    if source and source.startswith("file://"):
        model_dir = file_uri_to_path(source)
        print("Resolved model_dir from file:// source:", model_dir)

    # 2b) If source is runs:/<run_id>/artifacts/... resolve under this run
    elif source and source.startswith("runs:/"):
        try:
            suffix = source.split(run_id, 1)[1]  # e.g. '/artifacts/model'
        except Exception:
            suffix = ""
        cur.execute("SELECT artifact_uri FROM runs WHERE run_uuid = ? OR run_uuid = ?", (run_id, run_id))
        row2 = cur.fetchone()
        if not row2:
            conn.close()
            raise SystemExit(f"Could not resolve artifact_uri for run_id={run_id}")
        artifacts_root_uri = row2[0]
        artifacts_root = file_uri_to_path(artifacts_root_uri)
        candidate = os.path.join(artifacts_root, suffix.lstrip("/"))
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "MLmodel")):
            model_dir = candidate
            print("Resolved model_dir from runs:/ source:", model_dir)
        else:
            model_dir = find_model_dir_from_run_artifacts(artifacts_root)
            print("Resolved model_dir by scanning artifacts root:", model_dir)

    # 2c) If source is models:/<model_id>, resolve to registry copy:
    elif source and source.startswith("models:/"):
        # Example: 'models:/m-bd2c15...'
        model_obj_id = source.split("models:/", 1)[1].split("/", 1)[0]  # 'm-bd2c15...'
        # Get the run's artifacts root to derive the experiment root: .../mlruns/<exp_id>/
        cur.execute("SELECT artifact_uri FROM runs WHERE run_uuid = ? OR run_uuid = ?", (run_id, run_id))
        row2 = cur.fetchone()
        if not row2:
            conn.close()
            raise SystemExit(f"Could not resolve artifact_uri for run_id={run_id}")
        artifacts_root_uri = row2[0]
        artifacts_root = file_uri_to_path(artifacts_root_uri)
        # experiment_root = .../mlruns/<exp_id>
        experiment_root = os.path.dirname(os.path.dirname(artifacts_root))  # strip '/<run_id>/artifacts'
        # Registry shelf layout: <experiment_root>/models/<model_obj_id>/artifacts
        candidate = os.path.join(experiment_root, "models", model_obj_id, "artifacts")
        print("Trying registry candidate:", candidate)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "MLmodel")):
            model_dir = candidate
            print("Resolved model_dir from models:/ source:", model_dir)
        else:
            # As a fallback, still scan the run's artifacts (may help other layouts)
            model_dir = find_model_dir_from_run_artifacts(artifacts_root)
            print("Fallback: resolved model_dir by scanning artifacts root:", model_dir)

    # 3) Anything else → fallback scan
    else:
        cur.execute("SELECT artifact_uri FROM runs WHERE run_uuid = ? OR run_uuid = ?", (run_id, run_id))
        row2 = cur.fetchone()
        conn.close()
        if not row2:
            raise SystemExit(f"Could not resolve artifact_uri for run_id={run_id}")
        artifacts_root_uri = row2[0]
        artifacts_root = file_uri_to_path(artifacts_root_uri)
        print("artifacts_root:", artifacts_root)
        model_dir = find_model_dir_from_run_artifacts(artifacts_root)
        print("Resolved model_dir by scanning artifacts root:", model_dir)

    if args.model_dir:
        if not (os.path.isdir(args.model_dir) and os.path.isfile(os.path.join(args.model_dir, "MLmodel"))):
            raise SystemExit(f"--model-dir does not contain MLmodel: {args.model_dir}")
        model_dir = args.model_dir
        print("Using user-provided model_dir:", model_dir)



    # 4) Pack model_dir into model.tar.gz (MLflow expects MLmodel at archive root)
    out_tar = "model.tar.gz"
    if os.path.exists(out_tar):
        os.remove(out_tar)

    with tarfile.open(out_tar, "w:gz") as tar:
        for root, _, files in os.walk(model_dir):
            for f in files:
                full = os.path.join(root, f)
                arcname = os.path.relpath(full, start=model_dir)
                tar.add(full, arcname=arcname)
    print(f"Packed: {out_tar}")

    # 5) Upload to S3
    s3 = boto3.client("s3", region_name=args.region)
    parsed = urlparse(args.s3_uri)
    if parsed.scheme != "s3":
        raise SystemExit(f"Invalid S3 URI: {args.s3_uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3.upload_file(out_tar, bucket, key)
    s3_uri = f"s3://{bucket}/{key}"
    print("Uploaded to:", s3_uri)

    # 6) Emit a small manifest for the deploy step
    with open("export_manifest.txt", "w", encoding="utf-8") as f:
        f.write(f"MODEL_NAME={args.model_name}\n")
        f.write(f"VERSION={version}\n")
        f.write(f"S3_ARTIFACT={s3_uri}\n")
        f.write(f"STAGE={args.stage}\n")
    print("Wrote export_manifest.txt")

if __name__ == "__main__":
    main()
