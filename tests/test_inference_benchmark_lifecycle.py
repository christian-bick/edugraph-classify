from __future__ import annotations

import json
import os
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from edugraph_classify import cli, inference_benchmark_cli as benchmark_cli
from edugraph_classify import inference_benchmark_config as benchmark_config
from edugraph_classify import inference_benchmark_resources as resources
from edugraph_classify import inference_benchmark_worker as benchmark_worker
from edugraph_classify import checkpoint_storage as storage
from edugraph_classify.dataset import file_sha256
from edugraph_classify.learning_data import write_json
from test_runpod import Store, prepared


OBSERVED = {"gpus": [{"name": "NVIDIA A40", "memory_gib": 47.4}], "gpu_count": 1,
            "host_ram_bytes": 50_000_000_000, "host_ram_gb_decimal": 50.0, "cpu_count": 9.0}


def recipe(source_model):
    return {
        "run_id": "inference-benchmark-test",
        "source_model": source_model,
        "benchmark": {"cohort": "validation", "max_tokens": 512, "ubatch_size": 1024, "quantization": "Q4_K_M",
                      "concurrency": [1, 4], "warmup": 2, "repeats": 1,
                      "llama_cpp_commit": "c" * 40},
        "execution": {"cloud_type": "SECURE", "gpu_type": "NVIDIA A40", "gpu_count": 1,
                      "minimum_vram_gib": 45, "minimum_ram_gib": 48, "minimum_vcpus": 8,
                      "container_disk_gb": 30, "volume_gb": 140, "max_hours": 2,
                      "max_compute_hour_usd": 1.0, "completion_action": "terminate"},
        "pricing": {"compute_hour_usd": 0.49, "disk_hour_usd": 0.024,
                    "estimated_hours": 0.5, "estimated_cost_limit_usd": 1.0},
        "artifacts": {"root_uri": "gs://edugraph-classify/runpod", "credentials_secret": "edugraph-gcs"},
    }


@pytest.fixture
def case(prepared, tmp_path):
    export = tmp_path / "source-export"
    export.mkdir()
    (export / "adapter").mkdir()
    (export / "adapter" / "weights.safetensors").write_bytes(b"fake weights")
    shutil.copyfile(prepared.path, export / "manifest.json")
    for name in ("prompt.json", "schema.json", "closed_schema.json", "chat_template.jinja"):
        shutil.copyfile(prepared.run / name, export / name)
    shutil.copytree(prepared.run / "processor", export / "processor")
    write_json(export / "result.json", {"status": "completed", "run_id": prepared.config["run_id"],
                                        "selected_epoch": 1})
    archive = tmp_path / "model.tar"
    storage.pack_files(export, [p.relative_to(export).as_posix() for p in export.rglob("*") if p.is_file()], archive)
    source_model = {"uri": "gs://edugraph-classify/runpod/models/" + prepared.config["run_id"] +
                           "/" + file_sha256(archive) + ".tar",
                    "sha256": file_sha256(archive), "bytes": archive.stat().st_size}
    config = recipe(source_model)
    config["benchmark"]["max_tokens"] = prepared.config["evaluation"]["max_tokens"]
    config_path = tmp_path / "recipe.json"
    write_json(config_path, config)
    return NS(config=config, config_path=config_path, model_bundle=archive, prepared=prepared,
              run=tmp_path / "runs" / config["run_id"])


def test_prepare_copies_only_validation_images_and_pins_selected_export(case):
    config = benchmark_config.load_inference_recipe(case.config_path)
    manifest = benchmark_cli.prepare_inference_benchmark(config, case.prepared.path, case.model_bundle,
                                                          case.run, "b" * 40)
    verified, examples = benchmark_config.verify_inference_prepared(case.run / "manifest.json", "b" * 40)
    assert verified == manifest
    assert len(examples) == case.prepared.manifest["cohort_counts"]["validation"]
    assert manifest["training_reference_count"] == case.prepared.manifest["cohort_counts"]["train"]
    assert all(row["split"] == "validation" for row in examples)
    assert len(list((case.run / "images").iterdir())) == len(examples)
    assert manifest["source_training"]["selected_epoch"] == 1
    assert manifest["source_training"]["recipe"]["dataset"] == case.prepared.config["dataset"]
    assert not (case.run / "adapter").exists()
    with pytest.raises(FileExistsError):
        benchmark_cli.prepare_inference_benchmark(config, case.prepared.path, case.model_bundle,
                                                  case.run, "b" * 40)
    image = case.run / examples[0]["image_path"]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        benchmark_config.verify_inference_prepared(case.run / "manifest.json", "b" * 40)


def test_prepare_rejects_changed_model_or_training_identity(case, tmp_path):
    bad = dict(case.config)
    bad["source_model"] = {**case.config["source_model"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="local model bundle"):
        benchmark_cli.prepare_inference_benchmark(bad, case.prepared.path, case.model_bundle,
                                                  tmp_path / "bad-digest", "b" * 40)
    bad = json.loads(json.dumps(case.config))
    bad["benchmark"]["max_tokens"] += 1
    with pytest.raises(ValueError, match="token ceiling"):
        benchmark_cli.prepare_inference_benchmark(bad, case.prepared.path, case.model_bundle,
                                                  tmp_path / "bad-tokens", "b" * 40)


@pytest.mark.parametrize("section,key,value", [
    ("benchmark", "cohort", "final_validation"),
    ("benchmark", "concurrency", [4, 1]),
    ("benchmark", "concurrency", [1, 1]),
    ("benchmark", "concurrency", []),
    ("benchmark", "llama_cpp_commit", "main"),
    ("benchmark", "max_tokens", 0),
    ("benchmark", "ubatch_size", 512),
    ("benchmark", "ubatch_size", 1030),
    ("benchmark", "warmup", -1),
    ("benchmark", "repeats", 0),
    ("execution", "cloud_type", "COMMUNITY"),
    ("execution", "gpu_count", 2),
    ("execution", "gpu_type", "bad/type"),
    ("execution", "minimum_ram_gib", 0),
    ("execution", "max_compute_hour_usd", float("nan")),
    ("execution", "completion_action", "keep"),
    ("pricing", "estimated_hours", 3),
    ("pricing", "estimated_cost_limit_usd", 0.01),
    ("source_model", "bytes", 0),
    ("source_model", "sha256", "bad"),
    ("source_model", "uri", "gs://other-bucket/model.tar"),
    ("artifacts", "credentials_secret", "secret/value"),
])
def test_bad_recipe_rejected(case, section, key, value):
    config = json.loads(json.dumps(case.config))
    config[section][key] = value
    with pytest.raises(ValueError):
        benchmark_config.validate_inference_recipe(config)


def test_recipe_rejects_unknown_fields_and_unpinned_source(case):
    config = json.loads(json.dumps(case.config))
    config["extra"] = "unsupported"
    with pytest.raises(ValueError, match="fields"):
        benchmark_config.validate_inference_recipe(config)
    config = json.loads(json.dumps(case.config))
    config["source_model"]["uri"] = "https://example.org/model.tar"
    with pytest.raises(ValueError, match="gs://"):
        benchmark_config.validate_inference_recipe(config)


def test_stage_render_launch_boundaries_and_secret_free_request(case, tmp_path, monkeypatch):
    benchmark_cli.prepare_inference_benchmark(case.config, case.prepared.path, case.model_bundle,
                                              case.run, "b" * 40)
    store = Store(tmp_path / "store")
    monkeypatch.setattr(benchmark_cli.GcsBundles, "from_environment", lambda: store)
    monkeypatch.setattr(benchmark_cli, "load_local_environment", lambda _: None)
    parser = cli.build_parser()
    check = parser.parse_args(["inference-benchmark", "check-config", "--config", str(case.config_path)])
    assert benchmark_cli.run_inference_benchmark_command(check, None)[1]["status"] == "valid"
    stage = parser.parse_args(["inference-benchmark", "stage", "--manifest", str(case.run / "manifest.json"),
                               "--confirm-run-id", case.config["run_id"]])
    with pytest.raises(ValueError, match="clean"):
        benchmark_cli.run_inference_benchmark_command(stage, None)
    stage.confirm_run_id = "wrong"
    with pytest.raises(ValueError, match="confirmation"):
        benchmark_cli.run_inference_benchmark_command(stage, "b" * 40)
    stage.confirm_run_id = case.config["run_id"]
    reference = benchmark_cli.run_inference_benchmark_command(stage, "b" * 40)[1]
    assert reference["manifest_identity"]
    image = "ghcr.io/owner/inference@sha256:" + "f" * 64
    render = parser.parse_args(["inference-benchmark", "render", "--manifest", str(case.run / "manifest.json"),
                                "--image", image])
    request = benchmark_cli.run_inference_benchmark_command(render, "b" * 40)[1]["request"]
    assert request["cloudType"] == "SECURE" and request["ports"] == []
    assert request["dockerEntrypoint"][-1] == "edugraph_classify.inference_benchmark_worker"
    assert request["env"]["EDUGRAPH_GCS_CREDENTIALS_JSON"].startswith("{{ RUNPOD_SECRET_")
    assert "WANDB_API_KEY" not in request["env"]
    assert "fake" not in json.dumps(request)
    with pytest.raises(ValueError, match="digest"):
        benchmark_config.pod_request(json.loads((case.run / "manifest.json").read_text()), "ghcr.io/owner/inference:latest", reference)
    provider = Mock()
    provider.list.return_value = []
    provider.create.return_value = {"id": "pod1", "name": case.config["run_id"]}
    provider.get.return_value = {"id": "pod1", "costPerHr": "0.49", "machine": {"secureCloud": True}}
    monkeypatch.setattr(benchmark_cli, "RunpodClient", lambda _: provider)
    monkeypatch.setenv("RUNPOD_API_KEY", "fake")
    launch = parser.parse_args(["inference-benchmark", "launch", "--manifest", str(case.run / "manifest.json"),
                                "--image", image, "--confirm-run-id", case.config["run_id"]])
    assert benchmark_cli.run_inference_benchmark_command(launch, "b" * 40)[1]["status"] == "created"
    assert provider.create.call_count == 1
    with pytest.raises(FileExistsError):
        benchmark_cli.run_inference_benchmark_command(launch, "b" * 40)
    stop = parser.parse_args(["inference-benchmark", "stop", "--launch-record", str(case.run / "pod-launch.json"),
                              "--confirm-run-id", case.config["run_id"]])
    stop.confirm_run_id = "wrong"
    with pytest.raises(ValueError, match="confirmation"):
        benchmark_cli.run_inference_benchmark_command(stop, None)
    stop.confirm_run_id = case.config["run_id"]
    benchmark_cli.run_inference_benchmark_command(stop, None)
    provider.stop.assert_called_once_with("pod1")
    status = parser.parse_args(["inference-benchmark", "status", "--launch-record", str(case.run / "pod-launch.json")])
    assert benchmark_cli.run_inference_benchmark_command(status, None)[1]["id"] == "pod1"


def test_cli_prepare_requires_clean_commit_and_completion_blocks_relaunch(case, tmp_path, monkeypatch):
    parser = cli.build_parser()
    prepare = parser.parse_args(["inference-benchmark", "prepare", "--config", str(case.config_path),
                                 "--training-manifest", str(case.prepared.path),
                                 "--model-bundle", str(case.model_bundle), "--runs-root", str(tmp_path / "new-runs")])
    with pytest.raises(ValueError, match="clean"):
        benchmark_cli.run_inference_benchmark_command(prepare, None)
    result = benchmark_cli.run_inference_benchmark_command(prepare, "b" * 40)[1]
    assert result["status"] == "prepared" and result["cohort_count"] == case.prepared.manifest["cohort_counts"]["validation"]
    monkeypatch.setattr(benchmark_cli, "load_local_environment", lambda _: None)
    store = Store(tmp_path / "store")
    monkeypatch.setattr(benchmark_cli.GcsBundles, "from_environment", lambda: store)
    manifest_path = Path(result["manifest"])
    stage = parser.parse_args(["inference-benchmark", "stage", "--manifest", str(manifest_path),
                               "--confirm-run-id", case.config["run_id"]])
    benchmark_cli.run_inference_benchmark_command(stage, "b" * 40)
    completed_uri = case.config["artifacts"]["root_uri"] + "/benchmarks/" + case.config["run_id"] + "/completed.json"
    store.complete({"run_id": case.config["run_id"]}, completed_uri)
    launch = parser.parse_args(["inference-benchmark", "launch", "--manifest", str(manifest_path),
                                "--image", "ghcr.io/owner/inference@sha256:" + "f" * 64,
                                "--confirm-run-id", case.config["run_id"]])
    with pytest.raises(ValueError, match="completed report"):
        benchmark_cli.run_inference_benchmark_command(launch, "b" * 40)


def test_worker_verifies_bundle_and_durable_result_before_termination(case, tmp_path, monkeypatch):
    benchmark_cli.prepare_inference_benchmark(case.config, case.prepared.path, case.model_bundle,
                                              case.run, "b" * 40)
    store = Store(tmp_path / "store")
    archive = tmp_path / "prepared.tar"
    manifest = json.loads((case.run / "manifest.json").read_text())
    storage.pack_files(case.run, ["manifest.json", *manifest["files_sha256"]], archive)
    ref = store.publish(archive, case.config["artifacts"]["root_uri"] + "/inputs", case.config["run_id"])
    for name, value in {
        "RUNPOD_POD_ID": "pod1", "RUNPOD_API_KEY": "pod-scoped-fake",
        "EDUGRAPH_MAX_HOURS": "2", "EDUGRAPH_EXPECTED_COMMIT": "b" * 40,
        "EDUGRAPH_CODE_COMMIT": "b" * 40, "EDUGRAPH_LLAMA_CPP_COMMIT": "c" * 40,
        "EDUGRAPH_BUNDLE_URI": ref["uri"], "EDUGRAPH_BUNDLE_SHA256": ref["sha256"],
        "EDUGRAPH_RUN_ID": case.config["run_id"],
    }.items():
        monkeypatch.setenv(name, value)
    hash_file = benchmark_worker.file_sha256
    monkeypatch.setattr(benchmark_worker, "file_sha256",
                        lambda path: manifest["uv_lock_sha256"] if Path(path) == Path("/app/uv.lock") else hash_file(path))
    monkeypatch.setattr(benchmark_worker.signal, "signal", Mock())
    client = Mock()
    client.get.return_value = {"id": "pod1", "costPerHr": "0.49", "machine": {"secureCloud": True}}
    timer = Mock()
    completion_uri = case.config["artifacts"]["root_uri"] + "/benchmarks/" + case.config["run_id"] + "/completed.json"

    def execute(path, work, artifact_store, **kwargs):
        assert path.name == "manifest.json" and path.parent.name == "prepared"
        assert kwargs["runtime_details"]["compute_hour_usd"] == 0.49
        artifact_store.complete({"run_id": case.config["run_id"]}, completion_uri)
        return {"status": "completed", "run_id": case.config["run_id"]}

    root = tmp_path / "worker"
    root.mkdir()
    result = benchmark_worker.run_worker(root, store_factory=lambda: store, client_factory=lambda _: client,
                                         execute=execute, timer_factory=Mock(return_value=timer),
                                         capacity_probe=lambda: OBSERVED)
    assert result["status"] == "completed"
    client.terminate.assert_called_once_with("pod1")
    timer.cancel.assert_called_once()

    root2 = tmp_path / "worker2"
    root2.mkdir()
    monkeypatch.setenv("EDUGRAPH_LLAMA_CPP_COMMIT", "wrong")
    client.reset_mock()
    with pytest.raises(ValueError, match="llama.cpp commit"):
        benchmark_worker.run_worker(root2, store_factory=lambda: store, client_factory=lambda _: client,
                                    execute=execute, timer_factory=Mock(return_value=Mock()),
                                    capacity_probe=lambda: OBSERVED)
    client.stop.assert_called_once_with("pod1")

    root3 = tmp_path / "worker3"
    root3.mkdir()
    monkeypatch.setenv("EDUGRAPH_LLAMA_CPP_COMMIT", "c" * 40)
    client.reset_mock()
    never_execute = Mock()
    markers = len(store.completions)
    with pytest.raises(ValueError, match="GPU type"):
        benchmark_worker.run_worker(root3, store_factory=lambda: store, client_factory=lambda _: client,
                                    execute=never_execute, timer_factory=Mock(return_value=Mock()),
                                    capacity_probe=lambda: {**OBSERVED, "gpus": [{"name": "NVIDIA L40S", "memory_gib": 48}]})
    assert len(store.completions) == markers + 1
    failure_uri = case.config["artifacts"]["root_uri"] + "/attempts/" + case.config["run_id"] + "/pod1/failure.json"
    assert store.read_json(failure_uri)["status"] == "failed"
    never_execute.assert_not_called()
    client.stop.assert_called_once_with("pod1")


def test_worker_does_not_terminate_without_completion_marker(case, tmp_path, monkeypatch):
    benchmark_cli.prepare_inference_benchmark(case.config, case.prepared.path, case.model_bundle,
                                              case.run, "b" * 40)
    store = Store(tmp_path / "store")
    archive = tmp_path / "prepared.tar"
    manifest = json.loads((case.run / "manifest.json").read_text())
    storage.pack_files(case.run, ["manifest.json", *manifest["files_sha256"]], archive)
    ref = store.publish(archive, case.config["artifacts"]["root_uri"] + "/inputs", case.config["run_id"])
    for name, value in {"RUNPOD_POD_ID": "pod1", "RUNPOD_API_KEY": "fake", "EDUGRAPH_MAX_HOURS": "2",
                        "EDUGRAPH_EXPECTED_COMMIT": "b" * 40, "EDUGRAPH_CODE_COMMIT": "b" * 40,
                        "EDUGRAPH_LLAMA_CPP_COMMIT": "c" * 40, "EDUGRAPH_BUNDLE_URI": ref["uri"],
                        "EDUGRAPH_BUNDLE_SHA256": ref["sha256"], "EDUGRAPH_RUN_ID": case.config["run_id"]}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(benchmark_worker, "file_sha256", lambda _: manifest["uv_lock_sha256"])
    monkeypatch.setattr(benchmark_worker.signal, "signal", Mock())
    client = Mock()
    client.get.return_value = {"costPerHr": "0.49", "machine": {"secureCloud": True}}
    root = tmp_path / "worker"
    root.mkdir()
    with pytest.raises(ValueError, match="durable completion marker"):
        benchmark_worker.run_worker(root, store_factory=lambda: store, client_factory=lambda _: client,
                                    execute=lambda *a, **k: {"status": "completed"},
                                    timer_factory=Mock(return_value=Mock()), capacity_probe=lambda: OBSERVED)
    client.stop.assert_called_once_with("pod1")
    client.terminate.assert_not_called()


def test_worker_main_sanitizes_provider_errors(monkeypatch, capsys):
    from edugraph_classify.providers.runpod import RunpodError

    def failed():
        raise RunpodError("secret value", status_code=403)

    monkeypatch.setattr(benchmark_worker, "run_worker", failed)
    assert benchmark_worker.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["errors"][0]["http_status"] == 403
    assert "secret value" not in json.dumps(report)
    monkeypatch.setattr(benchmark_worker, "run_worker", lambda: {"status": "completed"})
    assert benchmark_worker.main() == 0


def test_early_preflight_failure_is_logged_and_marked_before_stop(failure_worker, monkeypatch):
    case = failure_worker
    case.store.download = Mock(side_effect=RuntimeError("SECRET_DOWNLOAD_DETAIL"))
    run_id = case.case.config["run_id"]
    root_uri = case.case.config["artifacts"]["root_uri"]
    digest = os.environ["EDUGRAPH_BUNDLE_SHA256"]
    monkeypatch.setenv("EDUGRAPH_BUNDLE_URI", f"{root_uri}/inputs/{run_id}/{digest}.tar")
    events = []
    monkeypatch.setattr(benchmark_worker, "print", lambda value, **_: events.append(("log", value)), raising=False)
    case.client.stop.side_effect = lambda _: events.append(("stop", None))

    with pytest.raises(RuntimeError, match="SECRET_DOWNLOAD_DETAIL"):
        benchmark_worker.run_worker(case.root, **case.kwargs)

    case.client.stop.assert_called_once_with("pod1")
    assert [event for event, _ in events] == ["log", "stop"]
    logged = json.loads(events[0][1])
    assert logged["status"] == "failed"
    assert logged["errors"][0]["error_type"] == "RuntimeError"
    assert "SECRET_DOWNLOAD_DETAIL" not in events[0][1]
    marker_uri = case.case.config["artifacts"]["root_uri"] + "/attempts/" + case.case.config["run_id"] + "/pod1/failure.json"
    marker = case.store.read_json(marker_uri)
    assert marker["kind"] == "inference_benchmark_preflight_failure_v1"
    assert marker["errors"][0]["error_type"] == "RuntimeError"
    assert "SECRET_DOWNLOAD_DETAIL" not in json.dumps(marker)


@pytest.fixture
def failure_worker(case, tmp_path, monkeypatch):
    benchmark_cli.prepare_inference_benchmark(case.config, case.prepared.path, case.model_bundle,
                                              case.run, "b" * 40)
    store = Store(tmp_path / "store")
    archive = tmp_path / "prepared.tar"
    manifest = json.loads((case.run / "manifest.json").read_text())
    storage.pack_files(case.run, ["manifest.json", *manifest["files_sha256"]], archive)
    ref = store.publish(archive, case.config["artifacts"]["root_uri"] + "/inputs", case.config["run_id"])
    for name, value in {"RUNPOD_POD_ID": "pod1", "RUNPOD_API_KEY": "fake", "EDUGRAPH_MAX_HOURS": "2",
                        "EDUGRAPH_EXPECTED_COMMIT": "b" * 40, "EDUGRAPH_CODE_COMMIT": "b" * 40,
                        "EDUGRAPH_LLAMA_CPP_COMMIT": "c" * 40, "EDUGRAPH_BUNDLE_URI": ref["uri"],
                        "EDUGRAPH_BUNDLE_SHA256": ref["sha256"], "EDUGRAPH_RUN_ID": case.config["run_id"]}.items():
        monkeypatch.setenv(name, value)
    hash_file = benchmark_worker.file_sha256
    monkeypatch.setattr(benchmark_worker, "file_sha256",
                        lambda path: manifest["uv_lock_sha256"] if Path(path) == Path("/app/uv.lock") else hash_file(path))
    monkeypatch.setattr(benchmark_worker.signal, "signal", Mock())
    client = Mock()
    client.get.return_value = {"costPerHr": "0.49", "machine": {"secureCloud": True}}
    root = tmp_path / "worker-failure"
    root.mkdir()
    return NS(case=case, store=store, client=client, root=root,
              kwargs={"store_factory": lambda: store, "client_factory": lambda _: client,
                      "timer_factory": Mock(return_value=Mock()), "capacity_probe": lambda: OBSERVED})


def test_worker_publishes_partial_rounds_and_sanitized_failure(failure_worker):
    case = failure_worker
    progress = {"run_id": case.case.config["run_id"], "phase": "serving", "settings": [
        {"concurrency": 1, "rounds": [{"predictions": [{"text": "raw model response"}]}]}]}

    def execute(_manifest, work, _store, **_):
        work.mkdir()
        write_json(work / "progress.json", progress)
        (work / "logs").mkdir()
        (work / "logs" / "llama-c1.log").write_text("data:image/jpeg;base64,SECRET_IMAGE")
        raise RuntimeError("SECRET_EXCEPTION_VALUE")

    with pytest.raises(RuntimeError, match="SECRET_EXCEPTION_VALUE"):
        benchmark_worker.run_worker(case.root, execute=execute, **case.kwargs)
    case.client.stop.assert_called_once_with("pod1")
    marker_uri = case.case.config["artifacts"]["root_uri"] + "/attempts/" + case.case.config["run_id"] + "/pod1/failure.json"
    marker = case.store.read_json(marker_uri)
    assert marker["status"] == "failed" and marker["progress"]["status"] == "included"
    archive = case.store.root / (marker["failure"]["sha256"] + ".tar")
    with tarfile.open(archive) as contents:
        assert sorted(contents.getnames()) == ["failure.json", "progress.json"]
        failure = json.load(contents.extractfile("failure.json"))
        retained = json.load(contents.extractfile("progress.json"))
    assert retained == progress
    assert failure["errors"][0]["error_type"] == "RuntimeError"
    assert failure["errors"][0]["frames"]
    assert failure["server_logs"] == "omitted"
    assert "SECRET_EXCEPTION_VALUE" not in json.dumps(failure)
    assert "SECRET_IMAGE" not in archive.read_bytes().decode("latin1")


def test_verified_preflight_capacity_failure_publishes_bundle(failure_worker, monkeypatch):
    case = failure_worker
    never_execute = Mock()
    events = []
    monkeypatch.setattr(benchmark_worker, "print", lambda value, **_: events.append(("log", value)), raising=False)
    case.client.stop.side_effect = lambda _: events.append(("stop", None))
    capacity = lambda: {**OBSERVED, "gpus": [{"name": "NVIDIA L40S", "memory_gib": 48}]}

    with pytest.raises(ValueError, match="GPU type"):
        benchmark_worker.run_worker(case.root, execute=never_execute,
                                    **{**case.kwargs, "capacity_probe": capacity})

    never_execute.assert_not_called()
    assert [event for event, _ in events] == ["log", "stop"]
    marker_uri = case.case.config["artifacts"]["root_uri"] + "/attempts/" + case.case.config["run_id"] + "/pod1/failure.json"
    marker = case.store.read_json(marker_uri)
    assert marker["status"] == "failed" and marker["progress"]["status"] == "absent"
    with tarfile.open(case.store.root / (marker["failure"]["sha256"] + ".tar")) as contents:
        assert contents.getnames() == ["failure.json"]
        failure = json.load(contents.extractfile("failure.json"))
    assert failure["errors"][0]["error_type"] == "ValueError"
    assert failure["server_logs"] == "omitted"


def test_failed_failure_publication_preserves_original_runner_error(failure_worker):
    case = failure_worker
    case.store.publish = Mock(side_effect=RuntimeError("publication failed"))

    def execute(*_args, **_kwargs):
        raise ValueError("original runner failure")

    with pytest.raises(ValueError, match="original runner failure"):
        benchmark_worker.run_worker(case.root, execute=execute, **case.kwargs)
    case.store.publish.assert_called_once()
    case.client.stop.assert_called_once_with("pod1")


def test_shutdown_error_does_not_hide_runner_error(failure_worker, capsys):
    case = failure_worker
    case.store.publish = Mock(side_effect=RuntimeError("publication unavailable"))
    case.client.stop.side_effect = RuntimeError("stop unavailable")

    def execute(*_args, **_kwargs):
        raise ValueError("original runner failure")

    with pytest.raises(ValueError, match="original runner failure"):
        benchmark_worker.run_worker(case.root, execute=execute, **case.kwargs)
    case.client.stop.assert_called_once_with("pod1")
    shutdown = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert shutdown["event"] == "shutdown_failed"
    assert shutdown["errors"][0]["error_type"] == "RuntimeError"
    assert "stop unavailable" not in json.dumps(shutdown)


def test_failure_report_omits_oversized_or_wrong_run_progress(failure_worker, monkeypatch, tmp_path):
    case = failure_worker
    work = tmp_path / "partial"
    work.mkdir()
    monkeypatch.setattr(benchmark_worker, "MAX_PROGRESS_BYTES", 64)
    (work / "progress.json").write_text("x" * 65)
    report = benchmark_worker.publish_failure(ValueError("private"),
                                              json.loads((case.case.run / "manifest.json").read_text()),
                                              work, case.root, "pod1", case.store)
    assert report["progress"]["status"] == "omitted_too_large"
    with tarfile.open(case.store.root / (report["failure"]["sha256"] + ".tar")) as contents:
        assert contents.getnames() == ["failure.json"]


@pytest.mark.parametrize("change, message", [
    ({"gpus": []}, "GPU count"),
    ({"gpus": [{"name": "NVIDIA L40S", "memory_gib": 48.0}]}, "GPU type"),
    ({"gpus": [{"name": "A40", "memory_gib": 44.9}]}, "GPU memory"),
    ({"host_ram_bytes": 47_999_999_999}, "RAM"),
    ({"cpu_count": 7.9}, "CPU allowance"),
])
def test_local_capacity_guard_rejects_mismatch(case, change, message):
    with pytest.raises(ValueError, match=message):
        resources.require_capacity({**OBSERVED, **change}, case.config["execution"])
    assert resources.require_capacity(OBSERVED, case.config["execution"]) == OBSERVED


@pytest.mark.parametrize("output", ["", "NVIDIA A40\n", "NVIDIA A40, bad\n", "NVIDIA A40, -1\n",
                                      "NVIDIA A40, NaN\n"])
def test_gpu_inventory_rejects_missing_or_invalid_memory(output):
    with pytest.raises(ValueError):
        resources.parse_gpu_inventory(output)
    assert resources.parse_gpu_inventory("NVIDIA A40, 48534\n")[0]["memory_gib"] > 47


def test_local_probe_respects_container_memory_and_cpu_limits(monkeypatch):
    process = NS(stdout="NVIDIA A40, 48534\n")
    subprocess_run = Mock(return_value=process)
    monkeypatch.setattr(resources.subprocess, "run", subprocess_run)
    monkeypatch.setattr(resources.os, "sysconf", lambda field: 100_000 if field == "SC_PHYS_PAGES" else 1_000_000,
                        raising=False)
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _: set(range(16)), raising=False)
    monkeypatch.setattr(resources, "_cgroup_memory_limit", lambda: 50_000_000_000)
    monkeypatch.setattr(resources, "_cgroup_cpu_quota", lambda: 8.5)
    observed = resources.local_resources()
    assert observed["gpu_count"] == 1
    assert observed["host_ram_gb_decimal"] == 50
    assert observed["cpu_count"] == 8.5
    assert subprocess_run.call_args.args[0][0] == "nvidia-smi"


def test_local_probe_rejects_unavailable_ram_or_cpu(monkeypatch):
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: NS(stdout="A40, 48000\n"))
    monkeypatch.setattr(resources.os, "sysconf", lambda _: 0, raising=False)
    with pytest.raises(ValueError, match="RAM"):
        resources.local_resources()
    monkeypatch.setattr(resources.os, "sysconf", lambda _: 100_000, raising=False)
    monkeypatch.setattr(resources, "_cgroup_memory_limit", lambda: None)
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _: set(), raising=False)
    with pytest.raises(ValueError, match="CPU affinity"):
        resources.local_resources()


def test_cgroup_limits_support_current_and_older_linux_layouts(monkeypatch):
    values = {"/sys/fs/cgroup/memory.max": "50000000000",
              "/sys/fs/cgroup/cpu.max": "850000 100000"}

    def read_text(path, **_):
        key = path.as_posix()
        if key not in values:
            raise OSError("not mounted")
        return values[key]

    monkeypatch.setattr(resources.Path, "read_text", read_text)
    assert resources._cgroup_memory_limit() == 50_000_000_000
    assert resources._cgroup_cpu_quota() == 8.5
    values = {"/sys/fs/cgroup/memory.max": "max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes": "48000000000",
              "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "800000",
              "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"}
    assert resources._cgroup_memory_limit() == 48_000_000_000
    assert resources._cgroup_cpu_quota() == 8
    values["/sys/fs/cgroup/cpu/cpu.cfs_quota_us"] = "-1"
    assert resources._cgroup_cpu_quota() is None


def test_invalid_cgroup_limits_fail_closed(monkeypatch):
    values = {"/sys/fs/cgroup/memory.max": "invalid", "/sys/fs/cgroup/cpu.max": "0 100000"}

    def read_text(path, **_):
        key = path.as_posix()
        if key not in values:
            raise OSError("not mounted")
        return values[key]

    monkeypatch.setattr(resources.Path, "read_text", read_text)
    with pytest.raises(ValueError, match="memory limit"):
        resources._cgroup_memory_limit()
    with pytest.raises(ValueError, match="CPU quota"):
        resources._cgroup_cpu_quota()
