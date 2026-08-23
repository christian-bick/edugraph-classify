"""Explicit read-only preflight entry points."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .configuration import load_local_environment
from .dataset import DatasetConversionError, file_sha256
from .preflight import EligibilityStatus, PreflightError
from .providers.fireworks import (
    FireworksTrainingProvider,
    TrainingLaunchError,
)
from .training import TrainingConfigError, launch_specs, prepare_run


def run_fireworks_preflight(
    hypothesis: str,
    env_file: str,
    *,
    client: Any | None = None,
) -> tuple[int, dict[str, object]]:
    load_local_environment(env_file)
    if client is None:
        from fireworks import Fireworks

        client = Fireworks()
    result = FireworksTrainingProvider(client).preflight_model(hypothesis)
    exit_code = 0 if result.status is EligibilityStatus.ELIGIBLE else 2
    return exit_code, result.to_mapping()


def _code_commit(repo_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise TrainingConfigError("run preparation requires a clean committed worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_prepare(
    config_path: str,
    runs_root: str,
    workers: int,
    *,
    repo_root: Path | None = None,
    code_commit: str | None = None,
) -> tuple[int, dict[str, object]]:
    root = (repo_root or Path.cwd()).resolve()
    commit = code_commit or _code_commit(root)
    prepared = prepare_run(
        Path(config_path),
        repo_root=root,
        runs_root=Path(runs_root),
        code_commit=commit,
        workers=workers,
    )
    return 0, prepared.to_mapping()


def _write_launch_record(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run_launch(
    manifest_path: str,
    confirmation: str,
    env_file: str,
    *,
    client: Any | None = None,
    repo_root: Path | None = None,
    code_commit: str | None = None,
) -> tuple[int, dict[str, object]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or confirmation != run_id:
        raise TrainingLaunchError("launch confirmation must exactly match the prepared run_id")
    prepared_commit = manifest.get("code_commit")
    current_commit = code_commit or _code_commit((repo_root or Path.cwd()).resolve())
    if not isinstance(prepared_commit, str) or current_commit != prepared_commit:
        raise TrainingLaunchError(
            "launch requires the same clean code commit recorded during preparation"
        )
    uploads, job_spec = launch_specs(path)

    load_local_environment(env_file)
    if client is None:
        from fireworks import Fireworks

        client = Fireworks()
    provider = FireworksTrainingProvider(client)
    preflight = provider.preflight_model(job_spec.base_model.rsplit("/", 1)[-1])
    if preflight.status is not EligibilityStatus.ELIGIBLE:
        raise TrainingLaunchError("base model failed the final live eligibility preflight")
    if preflight.capabilities.identity.model_id != job_spec.base_model:
        raise TrainingLaunchError("live preflight resolved a different base model")

    record_path = path.parent / "launch-record.json"
    manifest_sha256 = file_sha256(path)
    record: dict[str, object] | None = None
    if record_path.is_file():
        candidate = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(candidate, dict):
            raise TrainingLaunchError("launch record must be an object")
        if (
            candidate.get("run_id") != run_id
            or candidate.get("manifest_sha256") != manifest_sha256
        ):
            raise TrainingLaunchError("launch record does not match the prepared manifest")
        if candidate.get("status") == "launched":
            return 0, {**candidate, "launch_record_path": str(record_path)}
        record = candidate

    dataset_records = [] if record is None else record.get("datasets")
    if not isinstance(dataset_records, list) or any(
        not isinstance(item, dict) for item in dataset_records
    ):
        raise TrainingLaunchError("launch record datasets are invalid")
    completed = {
        item.get("name")
        for item in dataset_records
        if isinstance(item.get("name"), str)
    }
    allowed_existing = set(completed)
    if record is not None and isinstance(record.get("pending_dataset"), str):
        allowed_existing.add(record["pending_dataset"])
    pending_job = record is not None and record.get("status") == "job_launch_pending"
    job_resource = f"accounts/{job_spec.account_id}/supervisedFineTuningJobs/{job_spec.job_id}"
    if pending_job:
        allowed_existing.update({job_resource, job_spec.output_model_name})

    existing = provider.validate_launch_targets(
        uploads,
        job_spec,
        allowed_existing=frozenset(allowed_existing),
    )
    if record is None:
        record = {
            "run_id": run_id,
            "manifest_sha256": manifest_sha256,
            "status": "validated",
            "datasets": dataset_records,
        }
        _write_launch_record(record_path, record)

    for upload in uploads:
        if upload.resource_name in completed:
            continue
        record["status"] = "dataset_upload_pending"
        record["pending_dataset"] = upload.resource_name
        _write_launch_record(record_path, record)
        observed = provider.upload_dataset(
            upload,
            create=upload.resource_name not in existing,
        )
        dataset_records.append({"split": upload.split, **asdict(observed)})
        record["status"] = "datasets_uploading"
        record["datasets"] = dataset_records
        record.pop("pending_dataset", None)
        _write_launch_record(record_path, record)

    record["status"] = "job_launch_pending"
    _write_launch_record(record_path, record)
    if job_resource in existing:
        job = provider.get_supervised_fine_tuning_job(job_spec)
    elif job_spec.output_model_name in existing:
        raise TrainingLaunchError("output model exists without its recorded training job")
    else:
        job = provider.launch_supervised_fine_tuning(job_spec)
    record["status"] = "launched"
    record["job"] = job.to_mapping()
    _write_launch_record(record_path, record)
    return 0, {**record, "launch_record_path": str(record_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edugraph-classify")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="run read-only provider capability checks")
    targets = preflight.add_subparsers(dest="target", required=True)

    fireworks = targets.add_parser("fireworks", help="inspect a Fireworks model hypothesis")
    fireworks.add_argument("--hypothesis", default="Qwen3-VL-8B-Instruct")
    fireworks.add_argument("--env-file", default=".env")

    run = commands.add_parser("run", help="prepare or launch a pinned experiment")
    actions = run.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare", help="convert and validate provider-ready data")
    prepare.add_argument(
        "--config",
        default="experiments/edugraph-20260823-qwen3vl8b-sft-v1.json",
    )
    prepare.add_argument("--runs-root", default="runs")
    prepare.add_argument("--workers", type=int, default=8)

    launch = actions.add_parser("launch", help="upload data and launch one authorized SFT job")
    launch.add_argument("--manifest", required=True)
    launch.add_argument("--confirm-run-id", required=True)
    launch.add_argument("--env-file", default=".env")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            exit_code, payload = run_fireworks_preflight(args.hypothesis, args.env_file)
        elif args.action == "prepare":
            exit_code, payload = run_prepare(args.config, args.runs_root, args.workers)
        else:
            exit_code, payload = run_launch(
                args.manifest,
                args.confirm_run_id,
                args.env_file,
            )
    except (PreflightError, DatasetConversionError, TrainingConfigError, TrainingLaunchError) as error:
        print(json.dumps({"status": "error", "message": str(error)}, indent=2), file=sys.stderr)
        return 2
    except Exception as error:  # Provider SDK errors must never expose request configuration.
        payload = {
            "status": "error",
            "message": f"command failed with {type(error).__name__}",
        }
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return exit_code
