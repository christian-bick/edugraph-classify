from __future__ import annotations

from unittest.mock import Mock

import pytest

from edugraph_classify.providers.runpod import RunpodError
from edugraph_classify.runpod_status import inspect_run
from edugraph_classify.self_hosted_run import resume_identity


@pytest.fixture
def manifest():
    recipe = {"run_id": "release-test", "dataset": {}, "ontology_version": "0.31.0",
              "model": {}, "selection": {}, "evaluation": {}, "training": {"epochs": 2},
              "artifacts": {"root_uri": "gs://example/runpod"}}
    return {"run_id": recipe["run_id"], "recipe": recipe, "code_commit": "a" * 40,
            "uv_lock_sha256": "b" * 64, "runtime_versions": {}, "ontology_snapshot_sha256": "c" * 64,
            "language_target_audit": {}, "files_sha256": {}}


@pytest.fixture
def completion(manifest):
    def reference(kind, char):
        digest = char * 64
        return {"run_id": manifest["run_id"], "sha256": digest, "bytes": 20,
                "uri": f"gs://example/runpod/{kind}/release-test/{digest}.tar"}
    return {"status": "completed", "run_id": manifest["run_id"], "identity": resume_identity(manifest),
            "selected_epoch": 1, "epochs_completed": 2, "step": 19,
            "model": reference("models", "d"), "continuation": reference("checkpoints", "e"),
            "unrelated_private_field": "private-value"}


def store_for(completion=None, startup=None):
    store = Mock()
    store.read_json.side_effect = lambda uri: completion if uri.endswith("completed.json") else startup
    return store


def record():
    return {"run_id": "release-test", "pod": {"id": "pod1"}, "request": {"env": {"TOKEN": "private-value"}}}


def test_status_needs_no_provider_key_and_does_not_claim_verification(manifest, completion):
    store = store_for(completion)
    result = inspect_run(manifest, store)
    assert result["status"] == "completed"
    assert result["completion"]["selected_epoch"] == 1
    assert result["completion"]["step"] == 19
    assert result["completion"]["model"] == completion["model"]
    assert result["artifacts_verified"] is False
    assert result["provider"] == result["startup"] == {"state": "not_checked"}
    assert "private-value" not in str(result)
    assert "termination_verified" not in result and "spend" not in str(result)
    store.publish.assert_not_called()
    store.complete.assert_not_called()


def test_startup_and_present_provider_are_separate_observations(manifest):
    startup = {"pod": {"id": "pod1", "secureCloud": True, "name": "release-test",
                        "desiredStatus": "RUNNING", "env": {"TOKEN": "private-value"}},
               "compute_hour_usd": .49, "private": "private-value"}
    provider = Mock()
    provider.get.return_value = {"id": "pod1", "desiredStatus": "EXITED", "machine": {"secureCloud": True},
                                 "env": {"TOKEN": "private-value"}}
    store = store_for(startup=startup)
    result = inspect_run(manifest, store, provider, record())
    assert result["status"] == "awaiting_completion"
    assert result["startup"]["state"] == "present"
    assert result["startup"]["pod"]["secureCloud"] is True
    assert result["provider"]["pod"]["desiredStatus"] == "EXITED"
    assert "private-value" not in str(result)
    assert store.read_json.call_args_list[1].args == ("gs://example/runpod/attempts/release-test/pod1/started.json",)
    provider.get.assert_called_once_with("pod1")
    provider.stop.assert_not_called()
    provider.terminate.assert_not_called()
    provider.create.assert_not_called()


@pytest.mark.parametrize("error", [RunpodError("private-value", status_code=403),
                                   RunpodError("private-value", status_code=500),
                                   RunpodError("private-value"), RuntimeError("private-value")])
def test_provider_errors_never_look_like_termination(manifest, completion, error):
    provider = Mock()
    provider.get.side_effect = error
    result = inspect_run(manifest, store_for(completion), provider, record())
    assert result["status"] == "completed"
    assert result["provider"]["state"] == "unknown"
    assert result["provider"]["error_type"] == type(error).__name__
    assert "private-value" not in str(result)


def test_only_404_means_provider_absent(manifest):
    provider = Mock()
    provider.get.side_effect = RunpodError("private-value", status_code=404)
    result = inspect_run(manifest, store_for(), provider, record())
    assert result["status"] == "awaiting_completion"
    assert result["provider"] == {"state": "absent", "pod_id": "pod1", "http_status": 404}
    assert result["startup"]["state"] == "missing"


@pytest.mark.parametrize("value", [[], "private-value", 1])
def test_malformed_storage_marker_is_unknown(manifest, value):
    store = store_for(value)
    result = inspect_run(manifest, store)
    assert result["status"] == "unknown"
    assert result["publication"]["error_type"] == "ValueError"
    assert result["completion"] is None


def test_storage_failures_are_sanitized(manifest):
    store = Mock()
    store.read_json.side_effect = RuntimeError("private-value")
    result = inspect_run(manifest, store, launch_record=record())
    assert result["status"] == "unknown"
    assert result["startup"]["state"] == "unknown"
    assert "private-value" not in str(result)


@pytest.mark.parametrize("change", ["run_id", "identity", "reference"])
def test_completion_identity_mismatch_is_not_reported_completed(manifest, completion, change):
    if change == "reference":
        completion["model"]["uri"] = "gs://other/secret"
    else:
        completion[change] = "wrong"
    result = inspect_run(manifest, store_for(completion))
    assert result["status"] == "unknown"
    assert result["completion"] is None
    assert "gs://other" not in str(result)


@pytest.mark.parametrize("pod", [None, {}, {"id": "other"}])
def test_invalid_startup_and_provider_pod_do_not_count_as_observed(manifest, pod):
    provider = Mock()
    provider.get.return_value = pod
    result = inspect_run(manifest, store_for(startup={"pod": pod}), provider, record())
    assert result["provider"]["state"] == "unknown"
    assert result["startup"]["state"] == "unknown"


def test_pending_launch_journal_does_not_invent_a_pod(manifest):
    provider = Mock()
    result = inspect_run(manifest, store_for(), provider, {"run_id": "release-test", "status": "create_pending"})
    assert result["provider"]["state"] == "not_checked"
    assert result["startup"]["state"] == "not_checked"
    provider.get.assert_not_called()


@pytest.mark.parametrize("case", ["manifest_id", "recipe_id", "prefix", "record_id", "pod_id"])
def test_local_identity_is_validated_before_network(manifest, case):
    launch = record()
    if case == "manifest_id": manifest["run_id"] = "../other"
    if case == "recipe_id": manifest["recipe"]["run_id"] = "wrong"
    if case == "prefix": manifest["recipe"]["artifacts"]["root_uri"] = "gs://example/root/../other"
    if case == "record_id": launch["run_id"] = "wrong"
    if case == "pod_id": launch["pod"]["id"] = "../other"
    store = Mock()
    with pytest.raises(ValueError): inspect_run(manifest, store, launch_record=launch)
    store.read_json.assert_not_called()
