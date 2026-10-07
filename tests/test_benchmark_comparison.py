"""The same saved validation cohort and evaluator govern every model stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from edugraph_classify.benchmark_comparison import compare_validation
from edugraph_classify import cli, inference_benchmark_cli
from edugraph_classify.checkpoint_storage import pack_files
from edugraph_classify.dataset import file_sha256
from edugraph_classify.evaluation import evaluate
from edugraph_classify.learning_data import write_json
from edugraph_classify.ontology import OntologyCatalog


EMPTY = {"areas": [], "scopes": [], "abilities": []}
ADDITION = {"areas": ["Addition"], "scopes": [], "abilities": []}
SUBTRACTION = {"areas": ["Subtraction"], "scopes": [], "abilities": []}


def _prediction(row: dict, labels: dict) -> dict:
    return {"id": row["id"], "text": json.dumps(labels), "latency_seconds": 0.1,
            "image_sha256": row["image_sha256"]}


def _hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def _benchmark(root: Path, name: str, route: str, source_model: dict,
               source_training: dict, examples: list[dict], training: list[dict],
               predictions: list[dict], catalog: OntologyCatalog, merged: dict):
    root.mkdir()
    recipe = {"run_id": name, "source_model": source_model,
              "benchmark": {"route": route, "cohort": "validation", "max_tokens": 512,
                            "concurrency": [1], "warmup": 0, "repeats": 1},
              "execution": {"cloud_type": "SECURE", "gpu_type": "NVIDIA L40S", "gpu_count": 1,
                            "minimum_vram_gib": 44, "minimum_ram_gib": 120,
                            "minimum_vcpus": 12, "container_disk_gb": 30,
                            "volume_gb": 350, "max_hours": 5, "max_compute_hour_usd": 2,
                            "completion_action": "terminate"},
              "pricing": {"compute_hour_usd": 1, "disk_hour_usd": 0.05,
                          "estimated_hours": 1, "estimated_cost_limit_usd": 2},
              "artifacts": {"root_uri": "gs://bucket/root",
                            "credentials_secret": "edugraph-gcs"}}
    if route == "merged_bf16_hf":
        recipe["benchmark"]["expected_merged_files_digest"] = hashlib.sha256(
            json.dumps(merged, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    else:
        recipe["benchmark"].update(quantization="Q4_K_M", ubatch_size=1024,
                                   llama_cpp_commit="c" * 40)
    write_json(root / "recipe.json", recipe)
    write_json(root / "examples.json", examples)
    write_json(root / "training_examples.json", training)
    for row in examples:
        target = root / row["image_path"]
        target.parent.mkdir(exist_ok=True)
        Image.new("RGB", (2, 2), (1, 2, 3)).save(target)
        assert file_sha256(target) == row["image_sha256"]
    manifest = {"kind": "inference_benchmark_v1", "run_id": name,
                "code_commit": "b" * 40, "uv_lock_sha256": "d" * 64,
                "recipe": recipe, "source_model": source_model,
                "source_training": source_training, "cohort": "validation",
                "cohort_count": len(examples), "training_reference_count": len(training),
                "files_sha256": _hashes(root)}
    write_json(root / "manifest.json", manifest)
    digest = recipe["benchmark"].get("expected_merged_files_digest")
    report = {"kind": ("merged_bf16_inference_benchmark_result_v1" if digest else
                       "inference_benchmark_result_v1"),
              "run_id": name, "source_training": source_training,
              "source_model": source_model, "benchmark_recipe": recipe,
              "code_commit": manifest["code_commit"],
              "uv_lock_sha256": manifest["uv_lock_sha256"],
              "input_manifest_sha256": file_sha256(root / "manifest.json"),
              "cohort": "validation", "cohort_count": len(examples),
              "settings": [{"concurrency": 1, "rounds": [{"repeat": 1, "concurrency": 1,
                  "predictions": predictions,
                  "metrics": evaluate(examples, predictions, training, catalog)}]}]}
    if digest:
        report["merged_hf"] = {"files_sha256": merged, "files_digest": digest}
    else:
        report["export"] = {"merged_hf": {"files_sha256": merged}}
    report_path = root / "result.json"
    write_json(report_path, report)
    return root / "manifest.json", report_path


@pytest.fixture
def comparison_case(tmp_path):
    catalog = OntologyCatalog.load("0.30.0")
    training_root = tmp_path / "training"
    training_root.mkdir()
    image = training_root / "image.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(image)
    image_sha = file_sha256(image)
    examples = [
        {"id": "validation/one", "split": "validation", "solution": False,
         "gold": ADDITION, "image_path": "image.png", "image_sha256": image_sha},
        {"id": "validation/two", "split": "validation", "solution": True,
         "gold": SUBTRACTION, "image_path": "image.png", "image_sha256": image_sha},
        {"id": "train/one", "split": "train", "solution": False,
         "gold": ADDITION, "image_path": "image.png", "image_sha256": image_sha},
    ]
    training = [{"id": row["id"], "gold": row["gold"], "solution": row["solution"]}
                for row in examples if row["split"] == "train"]
    validation = examples[:2]
    recipe = {"ontology_version": "0.30.0"}
    write_json(training_root / "recipe.json", recipe)
    write_json(training_root / "examples.json", examples)
    source_manifest = {"run_id": "training-run", "code_commit": "a" * 40,
                       "uv_lock_sha256": "e" * 64, "recipe": recipe,
                       "ontology_snapshot_sha256": catalog.snapshot_sha256,
                       "files_sha256": _hashes(training_root)}
    source_manifest_path = training_root / "manifest.json"
    write_json(source_manifest_path, source_manifest)
    raw = [_prediction(validation[0], ADDITION), _prediction(validation[1], EMPTY)]
    source = tmp_path / "model"
    (source / "reports").mkdir(parents=True)
    raw_path = source / "reports" / "epoch-2-predictions.json"
    metrics_path = source / "reports" / "epoch-2-metrics.json"
    write_json(raw_path, raw)
    write_json(metrics_path, {"validation": evaluate(validation, raw, training, catalog)})
    bundle = tmp_path / "model.tar"
    pack_files(source, ["reports/epoch-2-predictions.json", "reports/epoch-2-metrics.json"], bundle)
    source_model = {"uri": "gs://bucket/root/models/training-run/" + file_sha256(bundle) + ".tar",
                    "sha256": file_sha256(bundle), "bytes": bundle.stat().st_size}
    source_training = {"run_id": "training-run", "manifest_sha256": file_sha256(source_manifest_path),
                       "code_commit": source_manifest["code_commit"],
                       "uv_lock_sha256": source_manifest["uv_lock_sha256"],
                       "ontology_snapshot_sha256": catalog.snapshot_sha256,
                       "recipe": recipe, "selected_epoch": 2}
    merged = {"model.safetensors": "f" * 64}
    bf16 = _benchmark(tmp_path / "bf16", "bf16-run", "merged_bf16_hf", source_model,
                      source_training, validation, training,
                      [_prediction(validation[0], ADDITION), _prediction(validation[1], SUBTRACTION)],
                      catalog, merged)
    gguf = _benchmark(tmp_path / "gguf", "gguf-run", "gguf_q4_k_m", source_model,
                      source_training, validation, training,
                      [_prediction(validation[0], EMPTY), _prediction(validation[1], SUBTRACTION)],
                      catalog, merged)
    return source_manifest_path, bundle, raw_path, metrics_path, [bf16, gguf]


def _compare(case):
    source_manifest, bundle, raw, metrics, benchmarks = case
    return compare_validation(source_manifest, bundle, 2, raw, metrics, benchmarks)


def test_paired_comparison_replays_all_three_stages(comparison_case):
    result = _compare(comparison_case)
    assert result["baseline"]["metrics"]["exact_count"] == 1
    assert [row["rounds"][0]["metrics"]["exact_count"] for row in result["benchmarks"]] == [2, 1]
    assert result["benchmarks"][0]["rounds"][0]["paired_to_nf4"]["counts"]["recovered"] == 1
    assert result["benchmarks"][1]["rounds"][0]["paired_to_nf4"]["counts"]["regressed"] == 1
    assert result["merged_checkpoint_match"] is True


@pytest.mark.parametrize("tamper,match", [
    ("metric", "reported validation metrics"),
    ("image_hash", "prediction image hashes"),
    ("merged_hash", "checkpoint hash"),
    ("bundle", "source model bundle"),
])
def test_paired_comparison_rejects_changed_evidence(comparison_case, tamper, match):
    source_manifest, bundle, _, _, benchmarks = comparison_case
    if tamper == "bundle":
        bundle.write_bytes(bundle.read_bytes() + b"changed")
    else:
        report_path = benchmarks[0][1]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if tamper == "metric":
            report["settings"][0]["rounds"][0]["metrics"]["explicit"]["f1"] = 0
        elif tamper == "image_hash":
            report["settings"][0]["rounds"][0]["predictions"][0]["image_sha256"] = "0" * 64
        else:
            report["merged_hf"]["files_sha256"]["model.safetensors"] = "0" * 64
        write_json(report_path, report)
    with pytest.raises(ValueError, match=match):
        _compare(comparison_case)


@pytest.mark.parametrize("tamper,match", [
    ("source_epoch", "source training"),
    ("validation_gold", "validation IDs, gold labels"),
    ("training_reference", "training-frequency reference"),
    ("source_model", "source model bundles differ"),
    ("report_identity", "report identity"),
    ("missing_setting", "concurrency-one setting"),
    ("missing_repeat", "missing or duplicate repeats"),
    ("duplicate_prediction", "duplicate validation ID"),
    ("merged_mismatch", "different merged checkpoints"),
])
def test_paired_comparison_rejects_unpaired_stages(comparison_case, tamper, match):
    _, _, _, _, benchmarks = comparison_case
    manifest_path, report_path = benchmarks[1 if tamper in ("source_model", "merged_mismatch") else 0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if tamper == "source_epoch":
        manifest["source_training"]["selected_epoch"] = 3
    elif tamper in ("validation_gold", "training_reference"):
        name = "examples.json" if tamper == "validation_gold" else "training_examples.json"
        path = manifest_path.parent / name
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows[0]["gold"] = EMPTY
        write_json(path, rows)
        manifest["files_sha256"][name] = file_sha256(path)
    elif tamper == "source_model":
        manifest["source_model"]["bytes"] += 1
        manifest["recipe"]["source_model"]["bytes"] += 1
        write_json(manifest_path.parent / "recipe.json", manifest["recipe"])
        manifest["files_sha256"]["recipe.json"] = file_sha256(manifest_path.parent / "recipe.json")
    elif tamper == "report_identity":
        report["code_commit"] = "0" * 40
    elif tamper == "missing_setting":
        report["settings"] = []
    elif tamper == "missing_repeat":
        report["settings"][0]["rounds"] = []
    elif tamper == "duplicate_prediction":
        predictions = report["settings"][0]["rounds"][0]["predictions"]
        predictions[1]["id"] = predictions[0]["id"]
    elif tamper == "merged_mismatch":
        report["export"]["merged_hf"]["files_sha256"]["model.safetensors"] = "0" * 64
    write_json(manifest_path, manifest)
    write_json(report_path, report)
    with pytest.raises(ValueError, match=match):
        _compare(comparison_case)


def test_paired_comparison_requires_selected_epoch_and_candidate(comparison_case):
    manifest, bundle, raw, metrics, benchmarks = comparison_case
    with pytest.raises(ValueError, match="selected epoch"):
        compare_validation(manifest, bundle, 0, raw, metrics, benchmarks)
    with pytest.raises(ValueError, match="at least one benchmark"):
        compare_validation(manifest, bundle, 2, raw, metrics, [])
    one = compare_validation(manifest, bundle, 2, raw, metrics, benchmarks[:1])
    assert len(one["benchmarks"]) == 1
    assert "merged_checkpoint_match" not in one


def test_compare_cli_is_local_and_limits_output_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parser = cli.build_parser()
    common = ["inference-benchmark", "compare", "--training-manifest", "source.json",
              "--source-model-bundle", "model.tar", "--selected-epoch", "2",
              "--training-predictions", "predictions.json", "--training-metrics", "metrics.json",
              "--benchmark-manifest", "benchmark.json", "--benchmark-result", "result.json"]
    calls = []
    monkeypatch.setattr(inference_benchmark_cli, "compare_validation",
                        lambda *args: calls.append(args) or {"cohort_count": 2, "benchmarks": [{}]})
    args = parser.parse_args([*common, "--output", "temp/paired.json"])
    status, payload = inference_benchmark_cli.run_inference_benchmark_command(args, None)
    assert status == 0 and payload["benchmark_count"] == 1 and len(calls) == 1
    assert (tmp_path / "temp" / "paired.json").is_file()
    args = parser.parse_args([*common, "--output", "outside.json"])
    with pytest.raises(ValueError, match="under reports/ or temp/"):
        inference_benchmark_cli.run_inference_benchmark_command(args, None)
    args = parser.parse_args([*common, "--benchmark-manifest", "extra.json",
                              "--output", "temp/paired.json"])
    with pytest.raises(ValueError, match="one report result"):
        inference_benchmark_cli.run_inference_benchmark_command(args, None)
