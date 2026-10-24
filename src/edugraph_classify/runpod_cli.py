"""Local preparation and explicit upload/paid-launch boundaries for Secure Pods."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import httpx

from .checkpoint_storage import GcsBundles, pack_files, unpack_verified
from .configuration import load_local_environment
from .learning_data import prepare_learning, verify_prepared, write_json
from .providers.runpod import RunpodClient, public_status, require_secure, checked_hourly_rate
from .runpod_config import RUNTIME_PACKAGES, load_runpod_recipe, pod_request, manifest_identity
from .self_hosted_run import verify_resume
from .runpod_recipe import new_release_recipe, write_new_recipe
from .runpod_results import retrieve_result, verify_retrieved
from .runpod_status import inspect_run

DEFAULT_RECIPE = "experiments/edugraph-20261024-qwen38-27b-runpod-k6-v1.json"


def add_runpod_parser(commands):
    parser = commands.add_parser("runpod", help="Secure QLoRA training with GCS and W&B")
    actions = parser.add_subparsers(dest="action", required=True)
    check = actions.add_parser("check-config", help="validate the local recipe; no provider calls")
    check.add_argument("--config", default=DEFAULT_RECIPE)
    new = actions.add_parser("new-run", help="copy the validated policy with new release pins and an explicit budget")
    new.add_argument("--from-config", default=DEFAULT_RECIPE)
    for name in ("run-id", "dataset-release", "dataset-revision", "ontology-version", "output"):
        new.add_argument("--" + name, required=True)
    for name in ("estimated-hours", "max-hours", "cost-limit-usd"):
        new.add_argument("--" + name, type=float, required=True)
    new.add_argument("--context-tokens", type=int)
    prepare = actions.add_parser("prepare", help="public downloads and local conversion only")
    prepare.add_argument("--config", required=True, help="pinned recipe with a new run ID")
    prepare.add_argument("--runs-root", default="runs")
    prepare.add_argument("--dry-run", action="store_true",
                         help="validate locally before committing; output cannot be staged or launched")
    for action in ("stage", "render", "launch"):
        sub = actions.add_parser(action)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--env-file", default=".env")
        if action != "stage":
            sub.add_argument("--image", required=True, help="public GHCR image@sha256:digest")
            sub.add_argument("--resume-reference", help="local checkpoint reference JSON from an earlier run")
            sub.add_argument("--attempt", help="fresh short attempt name when recovering the same run")
        if action != "render":
            sub.add_argument("--confirm-run-id", required=True)
    for action in ("status", "stop"):
        sub = actions.add_parser(action)
        sub.add_argument("--launch-record", required=action == "stop")
        sub.add_argument("--env-file", default=".env")
        if action == "status":
            sub.add_argument("--manifest", help="also inspect durable publication; works after Pod termination")
        if action == "stop":
            sub.add_argument("--confirm-run-id", required=True)
    retrieve = actions.add_parser("retrieve", help="download and verify completed bundles; no cloud writes")
    retrieve.add_argument("--manifest", required=True)
    retrieve.add_argument("--output", required=True, help="local result directory, normally under reports/")
    retrieve.add_argument("--env-file", default=".env")
    verify = actions.add_parser("verify-result", help="offline tensor, checkpoint and raw-metric verification")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--result-dir", required=True, help="directory returned by retrieve")


def prepare_runpod(config, run, commit):
    from huggingface_hub import snapshot_download

    model = config["model"]
    processor = Path("temp") / f"processor-{model['hf_revision']}"
    snapshot_download(model["hf_repository"], revision=model["hf_revision"], token=False,
        local_dir=processor, allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
            "vocab.json", "merges.txt", "*processor_config.json", "chat_template*"])
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        def fetch(url):
            response = client.get(url)
            response.raise_for_status()
            return response.content
        return prepare_learning(config, Path.cwd(), run, processor, commit, fetch,
            {"surface": "self_hosted_transformers", "gpu_fit_verified": False}, runtime_packages=RUNTIME_PACKAGES)


def launch_pod(manifest, request, run, client, *, attempt=None):
    if any(pod.get("name") == request["name"] or
           ((pod.get("name") == manifest["run_id"] or pod.get("name", "").startswith(manifest["run_id"] + "--"))
            and pod.get("desiredStatus") != "EXITED") for pod in client.list()):
        raise ValueError("a Pod with this run ID already exists; inspect it before creating another")
    record = {"run_id": manifest["run_id"], "status": "create_pending", "request": request}
    path = run / ("pod-launch" + ("-" + attempt if attempt else "") + ".json")
    # An ambiguous HTTP failure must never cause an automatic duplicate paid Pod.
    with path.open("x", encoding="utf-8") as output:
        json.dump(record, output)
    pod = client.create(request)
    record.update(status="created", pod=public_status(pod))
    write_json(path, record)
    guard = "allocation"
    try:
        pod = client.get(pod["id"])
        record["pod"] = public_status(pod)
        require_secure(pod)
        guard = "price"
        checked_hourly_rate(pod, manifest["recipe"]["execution"]["max_compute_hour_usd"])
    except (ValueError, RuntimeError):
        client.stop(record["pod"]["id"])
        record["status"] = "stopped_" + guard + "_guard"
        write_json(path, record)
        raise
    write_json(path, record)
    return record


def check_resume_reference(manifest, reference, run, store):
    """Read-only restore verification before creating a paid replacement Pod."""
    with tempfile.TemporaryDirectory(prefix="resume-check-", dir=run) as scratch:
        root = Path(scratch)
        store.download(reference, root / "snapshot.tar")
        unpack_verified(root / "snapshot.tar", reference["sha256"], root / "checkpoint")
        state = verify_resume(manifest, root / "checkpoint")
        if (state["run_id"] == manifest["run_id"] and state["phase"] == "final" and
                store.read_json(manifest["recipe"]["artifacts"]["root_uri"] + "/runs/" + manifest["run_id"] + "/completed.json")):
            raise ValueError("this run already has a completed model; no replacement Pod is needed")
        return state


def run_runpod_command(args, code_commit):
    if args.action == "new-run":
        config = new_release_recipe(load_runpod_recipe(Path(args.from_config)),
            run_id=args.run_id, dataset_release=args.dataset_release, dataset_revision=args.dataset_revision,
            ontology_version=args.ontology_version, estimated_hours=args.estimated_hours,
            max_hours=args.max_hours, cost_limit_usd=args.cost_limit_usd, context_tokens=args.context_tokens)
        write_new_recipe(Path(args.output), config)
        return 0, {"status": "recipe_created", "config": args.output, "run_id": config["run_id"],
                   "ontology_version": config["ontology_version"], "selection": config["selection"],
                   "next_step": "sync the pinned ontology dependency, then prepare from a clean commit"}
    if args.action == "verify-result":
        directory = Path(args.result_dir)
        report = verify_retrieved(Path(args.manifest), directory)
        path = directory / "verified-result.json"
        write_json(path, report)
        return 0, {"status": "verified", "run_id": report["run_id"], "report": str(path)}
    if args.action in ("check-config", "prepare"):
        config = load_runpod_recipe(Path(args.config))
        if args.action == "check-config":
            return 0, {"status": "valid", "run_id": config["run_id"], "execution": config["execution"],
                       "tracking": config["tracking"], "gpu_fit_verified": False}
        if not code_commit and not args.dry_run:
            raise ValueError("preparation requires a clean code commit")
        run = Path(args.runs_root) / config["run_id"]
        manifest = prepare_runpod(config, run, "uncommitted-dry-run" if args.dry_run else code_commit)
        return 0, {"status": "dry_run_prepared" if args.dry_run else "prepared",
                   "launchable": not args.dry_run, "manifest": str(run / "manifest.json"),
                   "estimated_run_cost_usd": manifest["estimated_run_cost_usd"]}
    load_local_environment(args.env_file)
    if args.action == "retrieve":
        result = retrieve_result(Path(args.manifest), Path(args.output), GcsBundles.from_environment())
        return 0, {"status": "retrieved", "run_id": result["run_id"],
                   "retrieval": str(Path(args.output) / "retrieval.json")}
    if args.action == "status" and args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        record = json.loads(Path(args.launch_record).read_text(encoding="utf-8")) if args.launch_record else None
        key = os.environ.get("RUNPOD_API_KEY")
        client = RunpodClient(key) if key and record else None
        return 0, inspect_run(manifest, GcsBundles.from_environment(), client, record)
    if args.action in ("status", "stop"):
        if not args.launch_record:
            raise ValueError("status requires --manifest or --launch-record")
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
    manifest, _ = verify_prepared(path, code_commit)
    run = path.parent
    if args.action != "render" and args.confirm_run_id != manifest["run_id"]:
        raise ValueError("confirmation differs from the prepared run ID")
    if args.action == "stage":
        archive = run / "prepared.tar"
        pack_files(run, ["manifest.json", *manifest["files_sha256"]], archive)
        reference = GcsBundles.from_environment().publish(archive,
            manifest["recipe"]["artifacts"]["root_uri"] + "/inputs/" + manifest["run_id"], manifest["run_id"])
        reference["manifest_identity"] = manifest_identity(manifest)
        write_json(run / "staged.json", reference)
        archive.unlink()
        return 0, reference
    staged = json.loads((run / "staged.json").read_text(encoding="utf-8"))
    resume = json.loads(Path(args.resume_reference).read_text(encoding="utf-8")) if args.resume_reference else None
    request = pod_request(manifest, args.image, staged, resume=resume, attempt=args.attempt)
    if args.action == "render":
        write_json(run / "pod-request.json", request)
        return 0, {"request_path": str(run / "pod-request.json"), "request": request}
    store = GcsBundles.from_environment()
    if store.read_json(manifest["recipe"]["artifacts"]["root_uri"] + "/runs/" + manifest["run_id"] + "/completed.json"):
        raise ValueError("this run already has a completed model; use a new run ID for continuation")
    if resume:
        check_resume_reference(manifest, resume, run, store)
    return 0, launch_pod(manifest, request, run, RunpodClient(os.environ["RUNPOD_API_KEY"]), attempt=args.attempt)
