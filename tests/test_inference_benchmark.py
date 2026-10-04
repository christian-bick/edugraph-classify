"""Offline tests of the GGUF benchmark's scoring and publication boundary."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from urllib.error import URLError

import pytest
from PIL import Image

from edugraph_classify.checkpoint_storage import pack_files
from edugraph_classify.dataset import file_sha256
import edugraph_classify.inference_benchmark as benchmark
from edugraph_classify.inference_benchmark import completion_request, execute_benchmark, jpeg_data_url
from edugraph_classify.learning_data import write_json
from edugraph_classify.ontology import OntologyCatalog


EMPTY_LABELS = {"areas": [], "scopes": [], "abilities": []}
SOURCE_COMMIT = "a" * 40
CODE_COMMIT = "b" * 40
LLAMA_COMMIT = "c" * 40


def prepared_fixture(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "images").mkdir()
    image = Image.new("RGB", (4, 3), (25, 50, 75))
    picture = prepared / "images" / "task.png"
    image.save(picture)
    example = {"id": "validation/task.png", "split": "validation", "solution": False,
               "gold": EMPTY_LABELS, "image_path": "images/task.png",
               "image_sha256": file_sha256(picture)}
    write_json(prepared / "examples.json", [example])
    write_json(prepared / "training_examples.json", [{"id": "train/task.png", "gold": EMPTY_LABELS,
                                                      "solution": False}])
    write_json(prepared / "prompt.json", {"system": "fixed", "user": "classify image"})
    write_json(prepared / "closed_schema.json", {"type": "object"})
    (prepared / "chat_template.jinja").write_text("fixed template", encoding="utf-8")

    source = tmp_path / "source"
    source.mkdir()
    source_recipe = {"model": {"hf_repository": "Qwen/Qwen3.8-27B", "hf_revision": SOURCE_COMMIT},
                     "ontology_version": "0.30.0", "training": {"max_context_tokens": 2048}}
    write_json(source / "manifest.json", {"run_id": "source-run", "recipe": source_recipe})
    write_json(source / "result.json", {"status": "completed", "selected_epoch": 2})
    archive = tmp_path / "source.tar"
    pack_files(source, ["manifest.json", "result.json"], archive)
    source_ref = {"uri": "gs://bucket/root/models/source-run/" + file_sha256(archive) + ".tar",
                  "sha256": file_sha256(archive), "bytes": archive.stat().st_size}
    recipe = {
        "run_id": "benchmark-run", "source_model": source_ref,
        "benchmark": {"cohort": "validation", "max_tokens": 512, "ubatch_size": 1024, "quantization": "Q4_K_M",
                      "concurrency": [1, 2], "warmup": 1, "repeats": 2,
                      "llama_cpp_commit": LLAMA_COMMIT},
        "execution": {"cloud_type": "SECURE", "gpu_type": "NVIDIA L40S", "gpu_count": 1,
                      "minimum_vram_gib": 45, "minimum_ram_gib": 120, "minimum_vcpus": 12,
                      "container_disk_gb": 30, "volume_gb": 350, "max_hours": 5,
                      "max_compute_hour_usd": 1.2, "completion_action": "terminate"},
        "pricing": {"compute_hour_usd": 1.09, "disk_hour_usd": 0.05,
                    "estimated_hours": 3, "estimated_cost_limit_usd": 5},
        "artifacts": {"root_uri": "gs://bucket/root", "credentials_secret": "edugraph-gcs"},
    }
    write_json(prepared / "recipe.json", recipe)
    manifest = {"kind": "inference_benchmark_v1", "run_id": "benchmark-run",
                "code_commit": CODE_COMMIT, "uv_lock_sha256": "d" * 64,
                "recipe": recipe, "source_model": source_ref,
                "source_training": {"run_id": "source-run", "manifest_sha256": file_sha256(source / "manifest.json"),
                                    "code_commit": SOURCE_COMMIT, "uv_lock_sha256": "e" * 64,
                                    "ontology_snapshot_sha256": OntologyCatalog.load("0.30.0").snapshot_sha256,
                                    "recipe": source_recipe, "selected_epoch": 2},
                "cohort": "validation", "cohort_count": 1, "training_reference_count": 1,
                "files_sha256": {path.relative_to(prepared).as_posix(): file_sha256(path)
                                 for path in prepared.rglob("*") if path.is_file()}}
    write_json(prepared / "manifest.json", manifest)
    return prepared / "manifest.json", archive


class FakeStore:
    def __init__(self, source: Path):
        self.source = source
        self.published = []
        self.markers = {}

    def download(self, reference, output):
        assert reference["sha256"] == file_sha256(self.source)
        shutil.copyfile(self.source, output)

    def publish(self, path, root_uri, run_id):
        names = []
        with tarfile.open(path) as packed:
            names = packed.getnames()
        result = {"run_id": run_id, "uri": root_uri + "/" + file_sha256(path) + ".tar",
                  "sha256": file_sha256(path), "bytes": path.stat().st_size}
        self.published.append((names, result))
        return result

    def complete(self, content, uri):
        self.markers[uri] = content


class FakeServer:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        assert kwargs["ubatch_size"] == 1024
        self.startup_seconds = 0.1

    def __enter__(self):
        self.calls.append(self.kwargs["parallel"])
        return self

    def __exit__(self, *_):
        return None

    def infer(self, request):
        assert request["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        return {"choices": [{"message": {"content": json.dumps(EMPTY_LABELS)},
                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class FakeSampler:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def summary(self):
        return {"peak_memory_mib": 12345, "sample_count": 1}


def fake_export(source, output, *, model_repo, model_revision, llama_dir, quantization):
    assert model_repo == "Qwen/Qwen3.8-27B" and model_revision == SOURCE_COMMIT
    assert quantization == "Q4_K_M"
    output.mkdir()
    files = ("model-Q4_K_M.gguf", "mmproj-BF16.gguf", "chat_template.jinja",
             "prompt.json", "closed_schema.json", "manifest.json", "export.json")
    for name in files:
        (output / name).write_bytes(name.encode())
    return {"gguf": {"llama_cpp_commit": LLAMA_COMMIT,
                     "language_quantized": {"path": str(output / files[0]),
                                            "sha256": file_sha256(output / files[0])},
                     "mmproj_bf16": {"path": str(output / files[1]),
                                     "sha256": file_sha256(output / files[1])}},
            "merged_hf": {"files_sha256": {"model.safetensors": "f" * 64}}}


def test_request_uses_fixed_template_schema_and_rgb_jpeg(tmp_path):
    image = tmp_path / "picture.png"
    Image.new("RGBA", (2, 2), (0, 50, 100, 80)).save(image)
    image_url = jpeg_data_url(image)
    assert image_url.startswith("data:image/jpeg;base64,")
    payload = completion_request(image_url, "classify", {"type": "object"}, 512)
    assert payload["messages"][0]["role"] == "user"
    assert payload["response_format"] == {"type": "json_object", "schema": {"type": "object"}}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0


def test_request_preserves_closed_schema_but_orders_dimensions_for_llama_cpp():
    schema = {"type": "object", "properties": {
        "abilities": {"type": "array", "items": {"type": "string", "enum": ["ProcedureExecution"]}},
        "areas": {"type": "array", "items": {"type": "string", "enum": ["Addition"]}},
        "scopes": {"type": "array", "items": {"type": "string", "enum": ["IntegerNumbers"]}},
    }, "required": ["areas", "scopes", "abilities"], "additionalProperties": False}

    payload = completion_request("data:image/jpeg;base64,AA==", "classify", schema, 512)
    sent = json.loads(json.dumps(payload))["response_format"]["schema"]

    assert list(sent["properties"]) == ["areas", "scopes", "abilities"]
    assert sent == schema
    assert list(schema["properties"]) == ["abilities", "areas", "scopes"]
    assert payload["response_format"]["schema"] is not schema
    assert payload["response_format"]["schema"]["properties"] is not schema["properties"]
    assert payload["response_format"]["schema"]["properties"]["areas"] is schema["properties"]["areas"]


def test_benchmark_scores_every_repeat_and_publishes_last(tmp_path):
    manifest, archive = prepared_fixture(tmp_path)
    store = FakeStore(archive)
    FakeServer.calls = []
    result = execute_benchmark(manifest, tmp_path / "work", store,
                               exporter=fake_export, server_factory=FakeServer,
                               sampler_factory=FakeSampler,
                               runtime_details={"compute_hour_usd": 1.09})
    assert result["status"] == "completed"
    assert FakeServer.calls == [1, 2]
    assert len(store.published) == 2
    assert store.published[0][0] == sorted(store.published[0][0])
    assert "model-Q4_K_M.gguf" in store.published[0][0]
    assert "mmproj-BF16.gguf" in store.published[0][0]
    report = json.loads((tmp_path / "work" / "report" / "result.json").read_text())
    assert len(report["settings"]) == 2
    assert all(len(setting["rounds"]) == 2 for setting in report["settings"])
    assert report["settings"][0]["rounds"][0]["metrics"]["explicit"]["exact_set_match"] == 1
    assert report["settings"][0]["rounds"][0]["metrics"]["usage"]["completion_tokens"] == 5
    assert report["settings"][0]["rounds"][0]["predictions"][0]["raw_response"]["choices"]
    prediction = report["settings"][0]["rounds"][0]["predictions"][0]
    assert prediction["latency_seconds"] >= prediction["server_latency_seconds"] >= 0
    assert prediction["preprocessing_seconds"] >= 0
    assert json.loads((tmp_path / "work" / "progress.json").read_text())["phase"] == "publishing"
    assert store.markers["gs://bucket/root/benchmarks/benchmark-run/completed.json"] == result


def test_no_completion_after_stop_or_export_mismatch(tmp_path):
    manifest, archive = prepared_fixture(tmp_path)
    store = FakeStore(archive)
    with pytest.raises(InterruptedError):
        execute_benchmark(manifest, tmp_path / "stopped", store, stop_requested=lambda: True,
                          exporter=fake_export, server_factory=FakeServer,
                          sampler_factory=FakeSampler)
    assert not store.markers and not store.published

    def wrong_export(*args, **kwargs):
        result = fake_export(*args, **kwargs)
        result["gguf"]["llama_cpp_commit"] = "0" * 40
        return result

    with pytest.raises(ValueError, match="converter revision"):
        execute_benchmark(manifest, tmp_path / "mismatch", store, exporter=wrong_export,
                          server_factory=FakeServer, sampler_factory=FakeSampler)
    assert not store.markers and not store.published


def test_tampered_validation_image_fails_before_scoring(tmp_path):
    manifest, archive = prepared_fixture(tmp_path)
    prepared = manifest.parent
    (prepared / "images" / "task.png").write_bytes(b"tampered")
    store = FakeStore(archive)
    with pytest.raises(ValueError, match="input hash"):
        execute_benchmark(manifest, tmp_path / "work", store, exporter=fake_export,
                          server_factory=FakeServer, sampler_factory=FakeSampler)
    assert not store.markers


def test_llama_server_starts_checks_health_and_posts_completion(tmp_path, monkeypatch):
    commands = []

    class Process:
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            assert timeout == 15

    process = Process()

    def start(command, **kwargs):
        commands.append(command)
        assert kwargs["stdout"].name.endswith("server.log")
        assert kwargs["stderr"] == subprocess.STDOUT
        return process

    class Response(io.BytesIO):
        status = 200

    requests = []

    def respond(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            raise URLError("loading")
        if isinstance(request, str):
            return Response(b'{"status":"ok"}')
        assert request.method == "POST"
        assert json.loads(request.data)["max_tokens"] == 512
        return Response(b'{"choices":[{"message":{"content":"{}"}}]}')

    monkeypatch.setattr(benchmark.subprocess, "Popen", start)
    monkeypatch.setattr(benchmark, "urlopen", respond)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _: None)
    with benchmark.LlamaServer(model=tmp_path / "model.gguf", mmproj=tmp_path / "mmproj.gguf",
                               chat_template=tmp_path / "chat.jinja", parallel=2,
                               context_per_slot=2048, ubatch_size=1024,
                               log_path=tmp_path / "server.log") as server:
        assert server.startup_seconds is not None
        assert server.infer({"max_tokens": 512})["choices"][0]["message"]["content"] == "{}"
    assert process.stopped
    assert "--parallel" in commands[0] and "--fit" in commands[0]
    assert commands[0].index("--jinja") < commands[0].index("--chat-template-file")
    assert commands[0][commands[0].index("--ctx-size") + 1] == "4096"
    assert commands[0][commands[0].index("--ubatch-size") + 1] == "1024"
    assert commands[0][commands[0].index("--n-gpu-layers") + 1] == "all"


@pytest.mark.parametrize("failure", ["exited", "stopped", "spawn"])
def test_llama_server_startup_failures_clean_up(tmp_path, monkeypatch, failure):
    class Process:
        terminated = False

        def poll(self):
            return 1 if failure == "exited" else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            pass

    process = Process()

    def start(*_, **__):
        if failure == "spawn":
            raise OSError("missing server")
        return process

    monkeypatch.setattr(benchmark.subprocess, "Popen", start)
    monkeypatch.setattr(benchmark, "urlopen", lambda *_, **__: pytest.fail("health should not be read"))
    server = benchmark.LlamaServer(model=tmp_path / "model.gguf", mmproj=tmp_path / "mmproj.gguf",
                                   chat_template=tmp_path / "chat.jinja", parallel=1,
                                   context_per_slot=2048, ubatch_size=1024,
                                   log_path=tmp_path / "server.log",
                                   stop_requested=lambda: failure == "stopped")
    with pytest.raises((RuntimeError, InterruptedError, OSError)):
        with server:
            pass
    assert process.terminated == (failure == "stopped")
    assert server.log.closed


def test_gpu_sampler_records_peak_and_survives_missing_nvidia_smi(monkeypatch):
    responses = iter(["120, 24000, 10", "220, 24000, 30"])

    def sample(*_, **__):
        try:
            return subprocess.CompletedProcess([], 0, next(responses))
        except StopIteration:
            raise FileNotFoundError("nvidia-smi missing")

    monkeypatch.setattr(benchmark.subprocess, "run", sample)
    sampler = benchmark.GpuSampler()
    with sampler:
        deadline = time.monotonic() + 3
        while len(sampler.samples) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
    assert sampler.summary()["peak_memory_mib"] == 220
    assert sampler.summary()["total_memory_mib"] == 24000
    assert sampler.summary()["mean_gpu_utilization_percent"] == 20


def test_predict_rejects_transport_shape_without_scoring(tmp_path):
    picture = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(picture)
    example = {"id": "task", "image_path": "image.png", "image_sha256": file_sha256(picture)}

    class InvalidServer:
        def infer(self, request):
            return {"choices": [{"message": {"content": ["not text"]}}]}

    with pytest.raises(ValueError, match="not text"):
        benchmark._predict(InvalidServer(), example, tmp_path, {"user": "classify"},
                           {"type": "object"}, 512, lambda: False)
    with pytest.raises(InterruptedError):
        benchmark._predict(InvalidServer(), example, tmp_path, {"user": "classify"},
                           {"type": "object"}, 512, lambda: True)
    example["image_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="image hash"):
        benchmark._predict(InvalidServer(), example, tmp_path, {"user": "classify"},
                           {"type": "object"}, 512, lambda: False)


@pytest.mark.parametrize("mismatch,reason", [
    ("ontology", "ontology snapshot"),
    ("size", "source model size"),
    ("training_manifest", "training manifest"),
    ("selected_epoch", "selected model epoch"),
    ("export_path", "outside its output directory"),
    ("export_hash", "export hash"),
])
def test_benchmark_rejects_changed_source_or_export_before_publication(tmp_path, mismatch, reason):
    manifest_path, archive = prepared_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    if mismatch == "ontology":
        manifest["source_training"]["ontology_snapshot_sha256"] = "0" * 64
    elif mismatch == "size":
        manifest["source_model"]["bytes"] += 1
        manifest["recipe"]["source_model"]["bytes"] += 1
    elif mismatch == "training_manifest":
        manifest["source_training"]["manifest_sha256"] = "0" * 64
    elif mismatch == "selected_epoch":
        manifest["source_training"]["selected_epoch"] = 1
    if mismatch == "size":
        write_json(manifest_path.parent / "recipe.json", manifest["recipe"])
        manifest["files_sha256"]["recipe.json"] = file_sha256(manifest_path.parent / "recipe.json")
    write_json(manifest_path, manifest)
    store = FakeStore(archive)

    def changed_export(*args, **kwargs):
        result = fake_export(*args, **kwargs)
        if mismatch == "export_path":
            result["gguf"]["language_quantized"]["path"] = str(archive)
        elif mismatch == "export_hash":
            result["gguf"]["language_quantized"]["sha256"] = "0" * 64
        return result

    with pytest.raises(ValueError, match=reason):
        execute_benchmark(manifest_path, tmp_path / "work", store, exporter=changed_export,
                          server_factory=FakeServer, sampler_factory=FakeSampler)
    assert not store.markers and not store.published


def test_report_upload_failure_cannot_mark_benchmark_complete(tmp_path):
    manifest, archive = prepared_fixture(tmp_path)

    class FailingStore(FakeStore):
        def publish(self, path, root_uri, run_id):
            if "/benchmarks/" in root_uri:
                raise OSError("report upload failed")
            return super().publish(path, root_uri, run_id)

    store = FailingStore(archive)
    with pytest.raises(OSError, match="report upload"):
        execute_benchmark(manifest, tmp_path / "work", store, exporter=fake_export,
                          server_factory=FakeServer, sampler_factory=FakeSampler)
    assert len(store.published) == 1
    assert not store.markers


def test_completed_rounds_survive_later_server_failure(tmp_path):
    manifest, archive = prepared_fixture(tmp_path)
    store = FakeStore(archive)

    class FailsSecondServer(FakeServer):
        def __enter__(self):
            if self.kwargs["parallel"] == 2:
                raise RuntimeError("server startup failed")
            return super().__enter__()

    with pytest.raises(RuntimeError, match="server startup"):
        execute_benchmark(manifest, tmp_path / "work", store, exporter=fake_export,
                          server_factory=FailsSecondServer, sampler_factory=FakeSampler)
    progress = json.loads((tmp_path / "work" / "progress.json").read_text())
    assert progress["phase"] == "serving"
    assert len(progress["settings"]) == 1
    assert len(progress["settings"][0]["rounds"]) == 2
    assert not store.markers


def test_llama_server_rejects_response_without_choice(monkeypatch):
    monkeypatch.setattr(benchmark, "urlopen", lambda *_, **__: io.BytesIO(b'{"choices":[]}'))
    server = benchmark.LlamaServer(model=Path("model.gguf"), mmproj=Path("mmproj.gguf"),
                                   chat_template=Path("chat.jinja"), parallel=1,
                                   context_per_slot=2048, ubatch_size=1024,
                                   log_path=Path("unused.log"))
    with pytest.raises(ValueError, match="lacks a completion choice"):
        server.infer({"max_tokens": 512})


def test_llama_server_kills_process_that_does_not_terminate(tmp_path):
    class SlowProcess:
        killed = False

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired("server", timeout)

        def kill(self):
            self.killed = True

    server = benchmark.LlamaServer(model=tmp_path / "model.gguf", mmproj=tmp_path / "mmproj.gguf",
                                   chat_template=tmp_path / "chat.jinja", parallel=1,
                                   context_per_slot=2048, ubatch_size=1024,
                                   log_path=tmp_path / "log")
    process = SlowProcess()
    server.process = process
    server.__exit__(None, None, None)
    assert process.killed
