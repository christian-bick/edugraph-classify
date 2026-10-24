"""Read-only publication and Pod observations for any prepared RunPod run."""
from __future__ import annotations

import re

from .gcs_artifacts import split_gcs_uri
from .providers.runpod import RunpodClient, RunpodError, public_status


def _failure(error: Exception) -> dict:
    """Provider exception values can contain credentials or request configuration."""
    return {"state": "unknown", "error_type": type(error).__name__,
            "http_status": error.status_code if isinstance(error, RunpodError) else None}


def _read_marker(store, uri: str) -> tuple[dict | None, dict]:
    try:
        marker = store.read_json(uri)
        if marker is not None and not isinstance(marker, dict):
            raise ValueError("run marker must be an object")
        return marker, {"state": "missing" if marker is None else "present", "uri": uri}
    except Exception as error:
        return None, {**_failure(error), "uri": uri}


def _pod_status(pod: dict) -> dict:
    result = public_status(pod)
    # The worker stores the already-sanitized, flat form in started.json.
    if "secureCloud" in pod:
        result["secureCloud"] = pod["secureCloud"]
    return result


def inspect_run(manifest: dict, store, client=None, launch_record: dict | None = None) -> dict:
    """Observe publication and one allocation without modifying either provider.

    Completion is an immutable publication marker, not independent tensor/metric
    verification. Startup is historical evidence, not a heartbeat. Only a REST
    404 establishes that the specified Pod is absent; other failures are unknown.
    """
    from .runpod_results import validate_completion

    run_id = manifest["run_id"]
    if (not isinstance(run_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", run_id)
            or manifest["recipe"]["run_id"] != run_id):
        raise ValueError("manifest run identity is invalid")
    root = manifest["recipe"]["artifacts"]["root_uri"].rstrip("/")
    _, prefix = split_gcs_uri(root)
    if ".." in prefix.split("/"):
        raise ValueError("artifact prefix cannot escape its root")
    if launch_record is not None and launch_record.get("run_id") != run_id:
        raise ValueError("launch record belongs to a different run")
    pod_id = (launch_record or {}).get("pod", {}).get("id")
    if pod_id is not None:
        RunpodClient.path(pod_id)

    completion, publication = _read_marker(store, f"{root}/runs/{run_id}/completed.json")
    if completion is not None:
        try:
            validate_completion(manifest, completion)
        except Exception as error:
            publication = {**_failure(error), "uri": publication["uri"]}
            completion = None
    summary = None if completion is None else {
        "run_id": run_id, "selected_epoch": completion["selected_epoch"],
        "epochs_completed": completion["epochs_completed"], "step": completion["step"],
        "model": {key: completion["model"][key] for key in ("run_id", "uri", "sha256", "bytes")},
        "continuation": {key: completion["continuation"][key] for key in ("run_id", "uri", "sha256", "bytes")},
    }

    startup, provider = {"state": "not_checked"}, {"state": "not_checked"}
    if pod_id is not None:
        marker, startup = _read_marker(store, f"{root}/attempts/{run_id}/{pod_id}/started.json")
        if marker is not None:
            if not isinstance(marker.get("pod"), dict) or marker["pod"].get("id") != pod_id:
                startup = {**_failure(ValueError()), "uri": startup["uri"]}
            else:
                startup["pod"] = _pod_status(marker["pod"])
        if client is not None:
            try:
                pod = client.get(pod_id)
                if not isinstance(pod, dict) or pod.get("id") != pod_id:
                    raise ValueError("provider returned a different Pod identity")
                provider = {"state": "present", "pod": _pod_status(pod)}
            except Exception as error:
                provider = ({"state": "absent", "pod_id": pod_id, "http_status": 404}
                            if isinstance(error, RunpodError) and error.status_code == 404
                            else _failure(error))

    return {"run_id": run_id,
            "status": "completed" if completion else
                      "unknown" if publication["state"] == "unknown" else "awaiting_completion",
            "publication": publication, "completion": summary, "artifacts_verified": False,
            "startup": startup, "provider": provider}
