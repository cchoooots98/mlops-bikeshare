import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import boto3
import mlflow

from src.model_package import PACKAGE_MANIFEST_FILENAME, load_package_manifest


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _copy_tree(item, target)
        else:
            target.write_bytes(item.read_bytes())


def _resolve_run_id(run_id: str | None, model_name: str | None, version: str | None, alias: str | None) -> str:
    if run_id:
        return run_id
    if not model_name:
        raise ValueError("provide --run-id or --model-name with --version/--alias")
    client = mlflow.tracking.MlflowClient()
    if version:
        return client.get_model_version(model_name, version).run_id
    if alias:
        return client.get_model_version_by_alias(model_name, alias).run_id
    raise ValueError("provide --version or --alias with --model-name")


def assemble_package_from_run(run_id: str, output_dir: Path) -> Path:
    package_dir = output_dir / run_id
    model_dir = package_dir / "model"
    artifacts_dir = package_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    downloaded_model = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model"))
    _copy_tree(downloaded_model, model_dir)

    manifest_artifact = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=PACKAGE_MANIFEST_FILENAME))
    package_manifest_path = package_dir / PACKAGE_MANIFEST_FILENAME
    package_manifest_path.write_bytes(manifest_artifact.read_bytes())

    for source_artifact, target_name in (
        ("eval/eval_summary.json", "eval_summary.json"),
        ("artifacts/model_card.md", "model_card.md"),
    ):
        try:
            artifact_path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=source_artifact))
        except Exception:
            continue
        (artifacts_dir / target_name).write_bytes(artifact_path.read_bytes())

    load_package_manifest(package_manifest_path)
    return package_dir


def create_package_tar(package_dir: Path, tar_path: Path) -> Path:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        for file_path in sorted(path for path in package_dir.rglob("*") if path.is_file()):
            tar.add(file_path, arcname=file_path.relative_to(package_dir).as_posix())
    return tar_path


def upload_to_s3(tar_path: Path, s3_uri: str, region: str) -> str:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"invalid S3 URI: {s3_uri}")
    boto3.client("s3", region_name=region).upload_file(str(tar_path), parsed.netloc, parsed.path.lstrip("/"))
    return s3_uri


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a model package and optionally upload it to S3.")
    parser.add_argument("--package-dir", default=None, help="Existing local model package directory.")
    parser.add_argument("--run-id", default=None, help="MLflow run id to export from.")
    parser.add_argument("--model-name", default=None, help="Registered model name.")
    parser.add_argument("--version", default=None, help="Registered model version to export.")
    parser.add_argument("--alias", default=None, help="Registered model alias to export.")
    parser.add_argument("--output-dir", default="dist/model_packages", help="Local directory for assembled packages.")
    parser.add_argument("--s3-uri", default=None, help="Optional destination S3 URI for the package tarball.")
    parser.add_argument("--region", default="eu-west-3", help="AWS region for optional S3 upload.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    if args.package_dir:
        package_dir = Path(args.package_dir).resolve()
        load_package_manifest(package_dir)
    else:
        run_id = _resolve_run_id(args.run_id, args.model_name, args.version, args.alias)
        with tempfile.TemporaryDirectory() as temp_dir:
            assembled_dir = assemble_package_from_run(run_id, Path(temp_dir))
            package_dir = output_dir / assembled_dir.name
            if package_dir.exists():
                shutil.rmtree(package_dir)
            _copy_tree(assembled_dir, package_dir)

    manifest = load_package_manifest(package_dir)
    tar_path = create_package_tar(package_dir, output_dir / f"{package_dir.name}.tar.gz")
    uploaded_uri = upload_to_s3(tar_path, args.s3_uri, args.region) if args.s3_uri else None
    result = {
        "package_dir": str(package_dir),
        "package_manifest_path": str(package_dir / PACKAGE_MANIFEST_FILENAME),
        "model_name": manifest["model_name"],
        "run_id": manifest["run_id"],
        "tar_path": str(tar_path),
        "s3_uri": uploaded_uri,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
