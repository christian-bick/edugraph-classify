"""Runpod entrypoint for the optional inference benchmark."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import traceback
from pathlib import Path

from .checkpoint_storage import GcsBundles, pack_files, unpack_verified
from .dataset import file_sha256
from .inference_benchmark_config import verify_inference_prepared
from .inference_benchmark_resources import local_resources, require_capacity
from .learning_data import write_json
from .providers.runpod import RunpodPodClient, RunpodError, checked_hourly_rate, public_status, require_secure


MAX_PROGRESS_BYTES = 50 * 1024 * 1024


def sanitized_errors(error: Exception) -> list[dict]:
    """Retain failure type and location without exception values or locals."""
    errors = []
    current = error
    while current is not None:
        errors.append({"error_type": type(current).__name__,
                       "http_status": current.status_code if isinstance(current, RunpodError) else None,
                       "frames": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                                  for frame in traceback.extract_tb(current.__traceback__)]})
        current = current.__cause__ or current.__context__
    return errors


def publish_failure(error: Exception, manifest: dict, work: Path, root: Path, pod_id: str, store) -> dict:
    """Preserve partial rounds and sanitized diagnostics before a failed Pod stops."""
    config = manifest["recipe"]
    attempt_uri = config["artifacts"]["root_uri"] + "/attempts/" + config["run_id"] + "/" + pod_id
    failure_dir = root / "failure"
    failure_dir.mkdir(parents=True, exist_ok=False)
    names = ["failure.json"]
    progress_note = {"status": "absent"}
    progress_path = work / "progress.json"
    if progress_path.is_file():
        with progress_path.open("rb") as source:
            content = source.read(MAX_PROGRESS_BYTES + 1)
        if len(content) > MAX_PROGRESS_BYTES:
            progress_note = {"status": "omitted_too_large", "limit_bytes": MAX_PROGRESS_BYTES}
        else:
            try:
                progress = json.loads(content)
            except (UnicodeDecodeError, json.JSONDecodeError):
                progress_note = {"status": "omitted_invalid_json"}
            else:
                if not isinstance(progress, dict) or progress.get("run_id") != config["run_id"]:
                    progress_note = {"status": "omitted_wrong_run"}
                else:
                    (failure_dir / "progress.json").write_bytes(content)
                    names.append("progress.json")
                    progress_note = {"status": "included", "bytes": len(content),
                                     "sha256": file_sha256(failure_dir / "progress.json")}
    report = {"kind": "inference_benchmark_failure_v1", "status": "failed",
              "run_id": config["run_id"], "pod_id": pod_id,
              "code_commit": manifest["code_commit"], "errors": sanitized_errors(error),
              "progress": progress_note, "server_logs": "omitted"}
    write_json(failure_dir / "failure.json", report)
    archive = root / "failure.tar"
    pack_files(failure_dir, names, archive)
    reference = store.publish(archive, attempt_uri, config["run_id"])
    marker = {"status": "failed", "run_id": config["run_id"], "failure": reference,
              "progress": progress_note}
    store.complete(marker, attempt_uri + "/failure.json")
    return marker


def run_worker(root=Path("/workspace"), *, store_factory=GcsBundles.from_environment,
               client_factory=RunpodPodClient, execute=None, timer_factory=threading.Timer,
               capacity_probe=local_resources):
    pod_id = os.environ["RUNPOD_POD_ID"]
    client = client_factory(os.environ["RUNPOD_API_KEY"])
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    timer = timer_factory(float(os.environ["EDUGRAPH_MAX_HOURS"]) * 3600, lambda: client.stop(pod_id))
    timer.daemon = True
    timer.start()
    completed, action = False, "stop"
    try:
        commit = os.environ["EDUGRAPH_EXPECTED_COMMIT"]
        if os.environ.get("EDUGRAPH_CODE_COMMIT") != commit:
            raise ValueError("container code commit differs from the benchmark bundle")
        pod = client.get(pod_id)
        require_secure(pod)
        store = store_factory()
        archive = root / "prepared.tar"
        reference = {"uri": os.environ["EDUGRAPH_BUNDLE_URI"], "sha256": os.environ["EDUGRAPH_BUNDLE_SHA256"]}
        store.download(reference, archive)
        prepared = root / "prepared"
        unpack_verified(archive, reference["sha256"], prepared)
        archive.unlink()
        manifest, _ = verify_inference_prepared(prepared / "manifest.json", commit)
        config = manifest["recipe"]
        if config["run_id"] != os.environ["EDUGRAPH_RUN_ID"] or file_sha256(Path("/app/uv.lock")) != manifest["uv_lock_sha256"]:
            raise ValueError("container lockfile or benchmark identity differs from preparation")
        if os.environ.get("EDUGRAPH_LLAMA_CPP_COMMIT") != config["benchmark"]["llama_cpp_commit"]:
            raise ValueError("container llama.cpp commit differs from the benchmark recipe")
        price = checked_hourly_rate(pod, config["execution"]["max_compute_hour_usd"])
        measured = require_capacity(capacity_probe(), config["execution"])
        runtime = {"pod": public_status(pod), "compute_hour_usd": price, "local_resources": measured}
        root_uri = config["artifacts"]["root_uri"]
        store.complete(runtime, root_uri + "/attempts/" + config["run_id"] + "/" + pod_id + "/started.json")
        if execute is None:
            from .inference_benchmark import execute_benchmark
            execute = execute_benchmark
        work = root / "benchmark"
        try:
            result = execute(prepared / "manifest.json", work, store,
                             stop_requested=stop.is_set, runtime_details=runtime)
        except Exception as error:
            try:
                publish_failure(error, manifest, work, root, pod_id, store)
            except BaseException:
                pass  # Failure publication must not hide the original runner error.
            raise
        completion_uri = root_uri + "/benchmarks/" + config["run_id"] + "/completed.json"
        marker = store.read_json(completion_uri)
        if result.get("status") == "completed" and (not isinstance(marker, dict) or marker.get("run_id") != config["run_id"]):
            raise ValueError("benchmark result lacks its durable completion marker")
        completed = result.get("status") == "completed"
        action = config["execution"]["completion_action"]
        print(json.dumps(result), flush=True)
        return result
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            timer.cancel()
            if completed and action == "terminate":
                client.terminate(pod_id)
            else:
                client.stop(pod_id)
        except Exception as shutdown_error:
            print(json.dumps({"event": "shutdown_failed", "pod_id": pod_id,
                              "errors": sanitized_errors(shutdown_error)}), flush=True)
            if not active_error:
                raise


def main():
    try:
        run_worker()
        return 0
    except Exception as error:
        # Remote failure diagnostics exclude exception values and local variables.
        print(json.dumps({"status": "failed", "errors": sanitized_errors(error)}), flush=True)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
