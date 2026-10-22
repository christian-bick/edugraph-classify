"""Runpod container entrypoint. No account-wide Runpod key is injected into Pods."""
from __future__ import annotations

import json
import os
import signal
import threading
import traceback
from pathlib import Path

from .checkpoint_storage import GcsBundles, unpack_verified
from .dataset import file_sha256
from .learning_data import verify_prepared
from .providers.runpod import RunpodPodClient, RunpodError, public_status, require_secure, checked_hourly_rate
from .runpod_config import validate_runpod_recipe
from .self_hosted_run import execute_self_hosted
from .training_tracking import WandbTracker


def run_worker(root=Path("/workspace"), *, store_factory=GcsBundles.from_environment,
               client_factory=RunpodPodClient, execute=execute_self_hosted, timer_factory=threading.Timer):
    pod_id = os.environ["RUNPOD_POD_ID"]
    client = client_factory(os.environ["RUNPOD_API_KEY"])  # Runpod supplies a Pod-scoped key.
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    timer = timer_factory(float(os.environ["EDUGRAPH_MAX_HOURS"]) * 3600, lambda: client.stop(pod_id))
    timer.daemon = True
    timer.start()
    completed, action = False, "stop"
    try:
        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError("WANDB_API_KEY secret is required")
        commit = os.environ["EDUGRAPH_EXPECTED_COMMIT"]
        if os.environ.get("EDUGRAPH_CODE_COMMIT") != commit:
            raise ValueError("container code commit differs from the prepared bundle")
        pod = client.get(pod_id)
        require_secure(pod)
        store = store_factory()
        archive = root / "prepared.tar"
        reference = {"uri": os.environ["EDUGRAPH_BUNDLE_URI"], "sha256": os.environ["EDUGRAPH_BUNDLE_SHA256"]}
        store.download(reference, archive)
        run = root / "prepared"
        unpack_verified(archive, reference["sha256"], run)
        archive.unlink()
        manifest, _ = verify_prepared(run / "manifest.json", commit)
        config = validate_runpod_recipe(manifest["recipe"])
        if config["run_id"] != os.environ["EDUGRAPH_RUN_ID"] or file_sha256(Path("/app/uv.lock")) != manifest["uv_lock_sha256"]:
            raise ValueError("container lockfile or run identity differs from preparation")
        price = checked_hourly_rate(pod, config["execution"]["max_compute_hour_usd"])
        runtime = {"pod": public_status(pod), "compute_hour_usd": price}
        # Check write permissions before model download or expensive validation.
        store.complete(runtime, config["artifacts"]["root_uri"] + "/attempts/" + config["run_id"] + "/" + pod_id + "/started.json")
        resume = None
        if os.environ.get("EDUGRAPH_RESUME_URI"):
            archive = root / "resume.tar"
            reference = {"uri": os.environ["EDUGRAPH_RESUME_URI"], "sha256": os.environ["EDUGRAPH_RESUME_SHA256"]}
            runtime["resume_reference"] = reference
            store.download(reference, archive)
            resume = root / "resume"
            unpack_verified(archive, reference["sha256"], resume)
            archive.unlink()
        from .self_hosted_training import load_qlora
        result = execute(run / "manifest.json", commit, root / "training", load_qlora, WandbTracker,
                         store, resume=resume, stop_requested=stop.is_set, runtime_details=runtime)
        completed = result["status"] == "completed"
        action = config["execution"]["completion_action"]
        print(json.dumps(result), flush=True)
        return result
    finally:
        timer.cancel()
        # Only discard the host's volume after verified external publication.
        if completed and action == "terminate":
            client.terminate(pod_id)
        else:
            client.stop(pod_id)


def main():
    try:
        run_worker()
        return 0
    except Exception as error:
        # Frame locations aid remote diagnosis without exposing exception values or locals.
        errors = []
        current = error
        while current is not None:
            errors.append({"error_type": type(current).__name__,
                "http_status": current.status_code if isinstance(current, RunpodError) else None,
                "frames": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                           for frame in traceback.extract_tb(current.__traceback__)]})
            current = current.__cause__ or current.__context__
        print(json.dumps({"status": "failed", "errors": errors}), flush=True)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
