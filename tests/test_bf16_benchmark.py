"""Offline checks of merged-BF16 scoring and report-only publication."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import edugraph_classify.bf16_benchmark as benchmark
from edugraph_classify.checkpoint_storage import pack_files
from edugraph_classify.dataset import file_sha256
from edugraph_classify.gguf_export import tree_sha256
from edugraph_classify.learning_data import write_json
from edugraph_classify.ontology import OntologyCatalog


EMPTY_LABELS = {"areas": [], "scopes": [], "abilities": []}
MERGED_CONTENT = {"config.json": b'{}', "model.safetensors": b"merged-bf16-weights"}
SOURCE_COMMIT = "a" * 40
CODE_COMMIT = "b" * 40


def merged_hashes() -> dict[str, str]:
    return {name: hashlib.sha256(content).hexdigest()
            for name, content in MERGED_CONTENT.items()}


def prepared_fixture(tmp_path: Path, *, expected_digest: str | None = None):
    prepared = tmp_path / "prepared"
    (prepared / "images").mkdir(parents=True)
    (prepared / "processor").mkdir()
    (prepared / "processor" / "tokenizer.json").write_text("{}", encoding="utf-8")
    picture = prepared / "images" / "task.png"
    Image.new("RGB", (4, 3), (25, 50, 75)).save(picture)
    example = {"id": "validation/task.png", "split": "validation", "solution": False,
               "gold": EMPTY_LABELS, "image_path": "images/task.png",
               "image_sha256": file_sha256(picture)}
    write_json(prepared / "examples.json", [example])
    write_json(prepared / "training_examples.json", [
        {"id": "train/task.png", "gold": EMPTY_LABELS, "solution": False}])
    write_json(prepared / "prompt.json", {"system": "fixed", "user": "classify image"})
    write_json(prepared / "closed_schema.json", {"type": "object"})

    source = tmp_path / "source"
    source.mkdir()
    source_recipe = {"model": {"hf_repository": "Qwen/Qwen3.8-27B",
                               "hf_revision": SOURCE_COMMIT},
                     "ontology_version": "0.30.0", "training": {"max_context_tokens": 2048}}
    write_json(source / "manifest.json", {"run_id": "source-run", "recipe": source_recipe})
    write_json(source / "result.json", {"status": "completed", "selected_epoch": 2})
    archive = tmp_path / "source.tar"
    pack_files(source, ["manifest.json", "result.json"], archive)
    source_ref = {"uri": "gs://bucket/root/models/source-run/" + file_sha256(archive) + ".tar",
                  "sha256": file_sha256(archive), "bytes": archive.stat().st_size}
    recipe = {
        "run_id": "benchmark-run", "source_model": source_ref,
        "benchmark": {"route": "merged_bf16_hf", "cohort": "validation", "max_tokens": 512,
                      "concurrency": [1], "warmup": 1, "repeats": 2,
                      "expected_merged_files_digest": expected_digest or
                          benchmark.merged_files_digest(merged_hashes())},
        "execution": {"cloud_type": "SECURE", "gpu_type": "NVIDIA A100 80GB PCIe",
                      "gpu_count": 1, "minimum_vram_gib": 78, "minimum_ram_gib": 120,
                      "minimum_vcpus": 12, "container_disk_gb": 30, "volume_gb": 250,
                      "max_hours": 5, "max_compute_hour_usd": 3.0,
                      "completion_action": "terminate"},
        "pricing": {"compute_hour_usd": 2.0, "disk_hour_usd": 0.05,
                    "estimated_hours": 2, "estimated_cost_limit_usd": 5},
        "artifacts": {"root_uri": "gs://bucket/root", "credentials_secret": "edugraph-gcs"},
    }
    write_json(prepared / "recipe.json", recipe)
    manifest = {
        "kind": "inference_benchmark_v1", "run_id": "benchmark-run",
        "code_commit": CODE_COMMIT, "uv_lock_sha256": "d" * 64,
        "recipe": recipe, "source_model": source_ref,
        "source_training": {"run_id": "source-run",
                            "manifest_sha256": file_sha256(source / "manifest.json"),
                            "code_commit": SOURCE_COMMIT, "uv_lock_sha256": "e" * 64,
                            "ontology_snapshot_sha256": OntologyCatalog.load("0.30.0").snapshot_sha256,
                            "recipe": source_recipe, "selected_epoch": 2},
        "cohort": "validation", "cohort_count": 1, "training_reference_count": 1,
        "files_sha256": {path.relative_to(prepared).as_posix(): file_sha256(path)
                         for path in prepared.rglob("*") if path.is_file()},
    }
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
        with tarfile.open(path) as packed:
            names = packed.getnames()
        reference = {"run_id": run_id, "uri": root_uri + "/" + file_sha256(path) + ".tar",
                     "sha256": file_sha256(path), "bytes": path.stat().st_size}
        self.published.append((names, reference))
        return reference

    def complete(self, content, uri):
        self.markers[uri] = content


class FakeSampler:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def summary(self):
        return {"peak_memory_mib": 54321, "sample_count": 1}


def fake_merge(source, merged_dir, *, model_repo, model_revision):
    assert (source / "result.json").is_file()
    assert model_repo == "Qwen/Qwen3.8-27B" and model_revision == SOURCE_COMMIT
    merged_dir.mkdir(parents=True)
    for name, content in MERGED_CONTENT.items():
        (merged_dir / name).write_bytes(content)
    return {"path": str(merged_dir), "files_sha256": tree_sha256(merged_dir)}


def fake_runtime(monkeypatch):
    calls = []
    renderer = SimpleNamespace(tokenizer=object())
    monkeypatch.setattr(benchmark.QwenVisionRenderer, "load", lambda *_: renderer)
    monkeypatch.setattr(benchmark, "render_prepared", lambda _run, _config, examples, _renderer:
                        {row["id"]: object() for row in examples})
    monkeypatch.setattr(benchmark, "tokenizer_constraints", lambda _tokenizer: object())

    def predict(_model, _renderer, _rendered, row, _schema, max_tokens, _constraints, _stop):
        assert max_tokens == 512
        calls.append(row["id"])
        return {"id": row["id"], "text": json.dumps(EMPTY_LABELS),
                "latency_seconds": 0.01, "generation_seconds": 0.008,
                "generated_token_ids": [1, 2], "finish_reason": "stop",
                "usage": {"prompt_tokens": 12, "completion_tokens": 2},
                "image_sha256": row["image_sha256"]}

    monkeypatch.setattr(benchmark, "predict_merged_bf16", predict)
    return calls


def test_merged_digest_is_stable_across_file_map_order():
    files = merged_hashes()
    assert benchmark.merged_files_digest(files) == benchmark.merged_files_digest(
        dict(reversed(list(files.items()))))
    assert benchmark.merged_files_digest(files) != benchmark.merged_files_digest(
        {**files, "config.json": "0" * 64})


def test_bf16_scores_each_repeat_and_publishes_only_report(tmp_path, monkeypatch):
    manifest, archive = prepared_fixture(tmp_path)
    calls = fake_runtime(monkeypatch)
    store = FakeStore(archive)
    result = benchmark.execute_bf16_benchmark(
        manifest, tmp_path / "work", store, merger=fake_merge,
        model_loader=lambda merged_dir: SimpleNamespace(path=merged_dir),
        sampler_factory=FakeSampler)

    assert result["status"] == "completed" and "candidate" not in result
    assert len(calls) == 3  # one warmup and two scored passes
    assert len(store.published) == 1 and store.published[0][0] == ["result.json"]
    assert store.markers["gs://bucket/root/benchmarks/benchmark-run/completed.json"] == result
    report = json.loads((tmp_path / "work" / "report" / "result.json").read_text())
    assert report["kind"] == "merged_bf16_inference_benchmark_result_v1"
    assert report["merged_hf"]["files_sha256"] == merged_hashes()
    assert report["merged_hf"]["files_digest"] == benchmark.merged_files_digest(merged_hashes())
    assert report["quality_promotion"] is False and report["cohort_count"] == 1
    assert len(report["settings"]) == 1 and len(report["settings"][0]["rounds"]) == 2
    for round_data in report["settings"][0]["rounds"]:
        assert round_data["metrics"]["explicit"]["exact_set_match"] == 1
        assert round_data["metrics"]["invalid_output_rate"] == 0
        assert round_data["predictions"][0]["generated_token_ids"] == [1, 2]
    assert not (tmp_path / "work" / "candidate.tar").exists()


def test_mismatched_merge_aborts_before_model_load_or_publication(tmp_path, monkeypatch):
    manifest, archive = prepared_fixture(tmp_path, expected_digest="0" * 64)
    fake_runtime(monkeypatch)
    store = FakeStore(archive)
    loaded = []
    with pytest.raises(ValueError, match="differs from the pinned v4 merge"):
        benchmark.execute_bf16_benchmark(
            manifest, tmp_path / "work", store, merger=fake_merge,
            model_loader=lambda path: loaded.append(path), sampler_factory=FakeSampler)
    assert not loaded and not store.published and not store.markers


def test_stop_and_changed_source_abort_before_gpu_work(tmp_path):
    manifest, archive = prepared_fixture(tmp_path)
    store = FakeStore(archive)
    with pytest.raises(InterruptedError, match="stop requested"):
        benchmark.execute_bf16_benchmark(
            manifest, tmp_path / "stopped", store, stop_requested=lambda: True,
            merger=fake_merge, model_loader=lambda _: pytest.fail("model loaded after stop"),
            sampler_factory=FakeSampler)
    assert not store.published and not store.markers

    class ChangedSourceStore(FakeStore):
        def download(self, reference, output):
            super().download(reference, output)
            output.write_bytes(b"changed")

    changed = ChangedSourceStore(archive)
    with pytest.raises(ValueError, match="source model size changed"):
        benchmark.execute_bf16_benchmark(
            manifest, tmp_path / "changed", changed, merger=fake_merge,
            model_loader=lambda _: pytest.fail("model loaded from changed source"),
            sampler_factory=FakeSampler)
    assert not changed.published and not changed.markers


@pytest.mark.parametrize("defect", ["path", "hash"])
def test_merged_checkpoint_attestation_precedes_model_loading(tmp_path, monkeypatch, defect):
    manifest, archive = prepared_fixture(tmp_path)
    fake_runtime(monkeypatch)
    store = FakeStore(archive)

    def bad_merge(*args, **kwargs):
        merged = fake_merge(*args, **kwargs)
        if defect == "path":
            merged["path"] = str(tmp_path / "unrelated-checkpoint")
        else:
            merged["files_sha256"]["model.safetensors"] = "0" * 64
        return merged

    with pytest.raises(ValueError, match="outside its output directory|hashes changed"):
        benchmark.execute_bf16_benchmark(
            manifest, tmp_path / "work", store, merger=bad_merge,
            model_loader=lambda _: pytest.fail("model loaded before merge attestation"),
            sampler_factory=FakeSampler)
    assert not store.published and not store.markers


def test_bf16_uses_training_generation_settings_and_checks_cuda(monkeypatch):
    import torch

    class BatchTensor:
        shape = (1, 3)

        def to(self, _device):
            return self

    class Completion:
        def tolist(self):
            return [7, 99]

    class Tokens:
        def __getitem__(self, index):
            assert index == (0, slice(3, None))
            return Completion()

    class Model:
        def __init__(self, device_type):
            self.device_type = device_type
            self.generation = None

        def parameters(self):
            yield SimpleNamespace(device=SimpleNamespace(type=self.device_type))

        def generate(self, **kwargs):
            self.generation = kwargs
            return Tokens()

    sentinel_constraint = object()
    monkeypatch.setattr(benchmark, "model_batch", lambda *_args, **_kwargs:
                        {"input_ids": BatchTensor()})
    monkeypatch.setattr(benchmark, "json_constraint", lambda *_args: sentinel_constraint)
    monkeypatch.setattr(torch, "inference_mode", lambda: nullcontext())
    monkeypatch.setattr(torch, "autocast", lambda *_args, **_kwargs: nullcontext())
    tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda ids, **kwargs:
                                json.dumps(EMPTY_LABELS) if ids == [7, 99] and
                                kwargs == {"skip_special_tokens": True} else "bad")
    renderer = SimpleNamespace(tokenizer=tokenizer, image_processor=object())
    row = {"id": "validation/task.png", "image_sha256": "f" * 64}
    model = Model("cuda")
    prediction = benchmark.predict_merged_bf16(
        model, renderer, SimpleNamespace(prompt_tokens=20), row, {}, 512,
        object(), lambda: False)

    assert prediction["text"] == json.dumps(EMPTY_LABELS)
    assert prediction["generated_token_ids"] == [7, 99]
    assert prediction["finish_reason"] == "stop"
    assert prediction["image_sha256"] == row["image_sha256"]
    assert prediction["usage"] == {"prompt_tokens": 20, "completion_tokens": 2}
    assert model.generation["max_new_tokens"] == 512
    assert model.generation["do_sample"] is False and model.generation["num_beams"] == 1
    assert model.generation["eos_token_id"] == model.generation["pad_token_id"] == 99
    assert model.generation["prefix_allowed_tokens_fn"] is sentinel_constraint

    with pytest.raises(ValueError, match="not fully resident on CUDA"):
        benchmark.predict_merged_bf16(
            Model("cpu"), renderer, SimpleNamespace(prompt_tokens=20), row, {}, 512,
            object(), lambda: False)
