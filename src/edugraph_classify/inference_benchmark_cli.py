"""Local preparation and explicit cloud boundaries for inference benchmarks."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .checkpoint_storage import GcsBundles, pack_files, unpack_verified
from .configuration import load_local_environment
from .dataset import file_sha256
from .inference_benchmark_config import (
    load_inference_recipe,
    manifest_identity,
    pod_request,
    verify_inference_prepared,
)
from .learning_data import safe_relative, verify_prepared, write_json
from .providers.runpod import RunpodClient, public_status
from .runpod_cli import launch_pod


def add_inference_benchmark_parser(commands):
    parser = commands.add_parser("inference-benchmark", help="optional Secure Runpod GGUF inference benchmark")
    actions = parser.add_subparsers(dest="action", required=True)
    check = actions.add_parser("check-config", help="validate a recipe without provider calls")
    check.add_argument("--config", required=True)
    prepare = actions.add_parser("prepare", help="verify selected model and prepare local validation inputs")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--training-manifest", required=True)
    prepare.add_argument("--model-bundle", required=True)
    prepare.add_argument("--runs-root", default="runs")
    for action in ("stage", "render", "launch"):
        sub = actions.add_parser(action)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--env-file", default=".env")
        if action != "stage":
            sub.add_argument("--image", required=True, help="public benchmark image@sha256:digest")
        if action != "render":
            sub.add_argument("--confirm-run-id", required=True)
    for action in ("status", "stop"):
        sub = actions.add_parser(action)
        sub.add_argument("--launch-record", required=True)
        sub.add_argument("--env-file", default=".env")
        if action == "stop":
            sub.add_argument("--confirm-run-id", required=True)


def prepare_inference_benchmark(config: dict, training_manifest_path: Path, model_bundle: Path,
                                run: Path, code_commit: str) -> dict:
    """Build a small, hashed evaluation bundle without provider writes or GPU work."""
    source_model = config["source_model"]
    if file_sha256(model_bundle) != source_model["sha256"] or model_bundle.stat().st_size != source_model["bytes"]:
        raise ValueError("local model bundle differs from the pinned source model")
    original = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    training_manifest, source_examples = verify_prepared(training_manifest_path, original["code_commit"])
    source_run_id = training_manifest["run_id"]
    if not source_model["uri"].endswith(f"/models/{source_run_id}/{source_model['sha256']}.tar"):
        raise ValueError("source model URI does not identify the verified training run")
    if config["benchmark"]["max_tokens"] != training_manifest["recipe"]["evaluation"]["max_tokens"]:
        raise ValueError("benchmark token ceiling must match the training evaluation contract")
    validation = [row for row in source_examples if row["split"] == "validation"]
    training = [row for row in source_examples if row["split"] == "train"]
    if (not validation or not training or
            len(validation) != training_manifest["cohort_counts"]["validation"] or
            len(training) != training_manifest["cohort_counts"]["train"]):
        raise ValueError("source training cohorts differ from the verified manifest")

    with tempfile.TemporaryDirectory(prefix="edugraph-model-verification-") as temporary:
        export = Path(temporary) / "export"
        unpack_verified(model_bundle, source_model["sha256"], export)
        if file_sha256(export / "manifest.json") != file_sha256(training_manifest_path):
            raise ValueError("model export does not match the supplied training manifest")
        result = json.loads((export / "result.json").read_text(encoding="utf-8"))
        if (result.get("status") != "completed" or result.get("run_id") != source_run_id or
                type(result.get("selected_epoch")) is not int or result["selected_epoch"] < 1):
            raise ValueError("source model is not a completed selected adapter export")
        if not (export / "adapter").is_dir() or not any((export / "adapter").iterdir()):
            raise ValueError("source model export lacks an adapter")
        required = ["prompt.json", "schema.json", "closed_schema.json", "chat_template.jinja"]
        processor_files = sorted(name for name in training_manifest["files_sha256"] if name.startswith("processor/"))
        if not processor_files:
            raise ValueError("source training manifest lacks processor files")
        for name in [*required, *processor_files]:
            if (name not in training_manifest["files_sha256"] or
                    file_sha256(export / safe_relative(name)) != training_manifest["files_sha256"][name]):
                raise ValueError("model export prompt, schema or processor differs from training")

        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "recipe.json", config)
        for name in [*required, *processor_files]:
            target = run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(export / name, target)
        for row in validation:
            image_path = safe_relative(row["image_path"])
            source = training_manifest_path.parent / image_path
            target = run / image_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if file_sha256(target) != row["image_sha256"]:
                raise ValueError("validation image bytes changed during benchmark preparation")
        write_json(run / "examples.json", validation)
        write_json(run / "training_examples.json", [
            {"id": row["id"], "gold": row["gold"], "solution": row["solution"]} for row in training
        ])

    source_training = {
        "run_id": source_run_id,
        "manifest_sha256": file_sha256(training_manifest_path),
        "code_commit": training_manifest["code_commit"],
        "uv_lock_sha256": training_manifest["uv_lock_sha256"],
        "ontology_snapshot_sha256": training_manifest["ontology_snapshot_sha256"],
        "recipe": training_manifest["recipe"],
        "selected_epoch": result["selected_epoch"],
    }
    manifest = {
        "kind": "inference_benchmark_v1",
        "run_id": config["run_id"], "code_commit": code_commit,
        "uv_lock_sha256": file_sha256(Path("uv.lock")),
        "recipe": config, "source_training": source_training, "source_model": source_model,
        "cohort": "validation", "cohort_count": len(validation), "training_reference_count": len(training),
        "estimated_run_cost_usd": config["pricing"]["estimated_hours"] *
                                  (config["pricing"]["compute_hour_usd"] + config["pricing"]["disk_hour_usd"]),
        "quality_promotion": False,
        "files_sha256": {p.relative_to(run).as_posix(): file_sha256(p)
                         for p in sorted(run.rglob("*")) if p.is_file()},
    }
    write_json(run / "manifest.json", manifest)
    verify_inference_prepared(run / "manifest.json", code_commit)
    return manifest


def run_inference_benchmark_command(args, code_commit):
    if args.action in ("check-config", "prepare"):
        config = load_inference_recipe(Path(args.config))
        if args.action == "check-config":
            return 0, {"status": "valid", "run_id": config["run_id"], "execution": config["execution"],
                       "estimated_cost_usd": config["pricing"]["estimated_hours"] *
                                               (config["pricing"]["compute_hour_usd"] + config["pricing"]["disk_hour_usd"])}
        if not code_commit:
            raise ValueError("benchmark preparation requires a clean code commit")
        run = Path(args.runs_root) / config["run_id"]
        manifest = prepare_inference_benchmark(config, Path(args.training_manifest), Path(args.model_bundle), run, code_commit)
        return 0, {"status": "prepared", "manifest": str(run / "manifest.json"),
                   "cohort_count": manifest["cohort_count"], "estimated_run_cost_usd": manifest["estimated_run_cost_usd"]}

    load_local_environment(args.env_file)
    if args.action in ("status", "stop"):
        record = json.loads(Path(args.launch_record).read_text(encoding="utf-8"))
        client = RunpodClient(os.environ["RUNPOD_API_KEY"])
        pod_id = record["pod"]["id"]
        if args.action == "stop":
            if args.confirm_run_id != record["run_id"]:
                raise ValueError("confirmation differs from the recorded run ID")
            client.stop(pod_id)
        return 0, public_status(client.get(pod_id))

    if not code_commit:
        raise ValueError("staging and launch rendering require a clean code commit")
    path = Path(args.manifest)
    manifest, _ = verify_inference_prepared(path, code_commit)
    run = path.parent
    if args.action != "render" and args.confirm_run_id != manifest["run_id"]:
        raise ValueError("confirmation differs from the prepared run ID")
    root_uri = manifest["recipe"]["artifacts"]["root_uri"]
    if args.action == "stage":
        archive = run / "prepared.tar"
        pack_files(run, ["manifest.json", *manifest["files_sha256"]], archive)
        try:
            reference = GcsBundles.from_environment().publish(
                archive, root_uri + "/inputs/" + manifest["run_id"], manifest["run_id"])
            reference["manifest_identity"] = manifest_identity(manifest)
            write_json(run / "staged.json", reference)
            return 0, reference
        finally:
            archive.unlink(missing_ok=True)

    staged = json.loads((run / "staged.json").read_text(encoding="utf-8"))
    request = pod_request(manifest, args.image, staged)
    if args.action == "render":
        write_json(run / "pod-request.json", request)
        return 0, {"request_path": str(run / "pod-request.json"), "request": request}
    store = GcsBundles.from_environment()
    if store.read_json(root_uri + "/benchmarks/" + manifest["run_id"] + "/completed.json"):
        raise ValueError("this benchmark already has a completed report; use a new run ID")
    return 0, launch_pod(manifest, request, run, RunpodClient(os.environ["RUNPOD_API_KEY"]))
