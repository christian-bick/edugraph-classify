"""Offline, paired validation comparison across training and inference routes."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from .dataset import file_sha256
from .evaluation import evaluate, label_pairs, parse_prediction
from .inference_benchmark_config import BF16_ROUTE, GGUF_ROUTE, verify_inference_prepared
from .learning_data import verify_prepared
from .ontology import OntologyCatalog


def _load(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_id(rows: list[dict]) -> dict[str, dict]:
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate validation ID")
    return indexed


def _checked_metrics(examples: list[dict], predictions: list[dict], training: list[dict],
                     catalog: OntologyCatalog, reported: dict) -> dict:
    measured = evaluate(examples, predictions, training, catalog)
    if measured != reported:
        raise ValueError("reported validation metrics differ from replayed predictions")
    return measured


def _exact_by_id(examples: list[dict], predictions: list[dict],
                 catalog: OntologyCatalog) -> dict[str, bool]:
    indexed = _by_id(predictions)
    return {
        row["id"]: (
            parsed["valid"] and parsed["canonical"] is not None and
            label_pairs(parsed["canonical"]) == label_pairs(row["gold"])
        )
        for row in examples
        for parsed in [parse_prediction(indexed[row["id"]]["text"], catalog)]
    }


def _summary(metrics: dict) -> dict:
    explicit = metrics["explicit"]
    return {
        "exact_count": round(explicit["exact_set_match"] * explicit["n"]),
        "cohort_count": explicit["n"],
        "exact_set_match": explicit["exact_set_match"],
        "precision": explicit["precision"],
        "recall": explicit["recall"],
        "f1": explicit["f1"],
        "invalid_count": round(metrics["invalid_output_rate"] * explicit["n"]),
        "per_dimension": metrics["per_dimension"],
        "per_view": metrics["per_view"],
    }


def _paired(examples: list[dict], baseline: list[dict], candidate: list[dict],
            catalog: OntologyCatalog) -> dict:
    old = _exact_by_id(examples, baseline, catalog)
    new = _exact_by_id(examples, candidate, catalog)
    groups = {name: [] for name in ("both_correct", "regressed", "recovered", "both_wrong")}
    for row in examples:
        key = row["id"]
        group = ("both_correct" if old[key] and new[key] else
                 "regressed" if old[key] else "recovered" if new[key] else "both_wrong")
        groups[group].append(key)
    return {"counts": {name: len(ids) for name, ids in groups.items()},
            "regressed_ids": groups["regressed"], "recovered_ids": groups["recovered"]}


def _copied_controls(files: dict[str, str]) -> set[str]:
    required = {"prompt.json", "schema.json", "closed_schema.json", "chat_template.jinja"}
    processor = {name for name in files if name.startswith("processor/")}
    if not required.issubset(files) or not processor:
        raise ValueError("original training manifest lacks prompt, schema, template or processor")
    return required | processor


def _verify_copied_inputs(original_files: dict[str, str], candidate_files: dict[str, str],
                          examples: list[dict]) -> None:
    controls = _copied_controls(original_files)
    if {name for name in candidate_files if name.startswith("processor/")} != {
            name for name in controls if name.startswith("processor/")}:
        raise ValueError("benchmark processor file set differs from training")
    shared = controls | {row["image_path"] for row in examples}
    for name in shared:
        if (not original_files.get(name) or
                candidate_files.get(name) != original_files[name]):
            raise ValueError(f"benchmark copied input differs from training: {name}")
    if any(row["image_sha256"] != original_files[row["image_path"]] for row in examples):
        raise ValueError("training validation image hash differs from its manifest")


def _archive_member(packed: tarfile.TarFile, name: str) -> bytes:
    try:
        member = packed.extractfile(name)
    except KeyError as error:
        raise ValueError(f"source model bundle lacks {name}") from error
    if member is None:
        raise ValueError(f"source model bundle lacks {name}")
    return member.read()


def _verify_source_bundle(bundle: Path, source_model: dict, training_manifest_path: Path,
                          training_manifest: dict, selected_epoch: int,
                          predictions_path: Path, metrics_path: Path,
                          baseline_metrics: dict) -> None:
    if bundle.stat().st_size != source_model["bytes"] or file_sha256(bundle) != source_model["sha256"]:
        raise ValueError("source model bundle differs from its pinned reference")
    with tarfile.open(bundle, "r") as packed:
        if _archive_member(packed, "manifest.json") != training_manifest_path.read_bytes():
            raise ValueError("source model bundle contains a different training manifest")
        result = json.loads(_archive_member(packed, "result.json"))
        if (result.get("status") != "completed" or
                result.get("run_id") != training_manifest["run_id"] or
                result.get("selected_epoch") != selected_epoch or
                result.get("selected_validation") != baseline_metrics["explicit"]):
            raise ValueError("source model bundle selected result differs from replayed validation")
        for name in _copied_controls(training_manifest["files_sha256"]):
            if hashlib.sha256(_archive_member(packed, name)).hexdigest() != \
                    training_manifest["files_sha256"][name]:
                raise ValueError(f"source model bundle differs at {name}")
        for path, member_name in (
            (predictions_path, f"reports/epoch-{selected_epoch}-predictions.json"),
            (metrics_path, f"reports/epoch-{selected_epoch}-metrics.json"),
        ):
            if _archive_member(packed, member_name) != path.read_bytes():
                raise ValueError(f"source model bundle differs at {member_name}")


def _merged_hashes(report: dict, route: str) -> dict[str, str]:
    return (report["merged_hf"] if route == BF16_ROUTE else report["export"]["merged_hf"])["files_sha256"]


def compare_validation(
    training_manifest_path: Path, source_model_bundle: Path, selected_epoch: int,
    training_predictions_path: Path, training_metrics_path: Path,
    benchmark_paths: list[tuple[Path, Path]],
) -> dict:
    """Verify identical inputs, replay metrics, and pair each concurrency-one round.

    Report archives must first be downloaded and checked against their immutable
    completion-marker SHA-256. This function performs no provider calls.
    """
    if type(selected_epoch) is not int or selected_epoch < 1 or not benchmark_paths:
        raise ValueError("a selected epoch and at least one benchmark are required")
    original = _load(training_manifest_path)
    training_manifest, all_examples = verify_prepared(
        training_manifest_path, original["code_commit"])
    examples = [row for row in all_examples if row["split"] == "validation"]
    training = [row for row in all_examples if row["split"] == "train"]
    if not examples or not training:
        raise ValueError("original training manifest lacks validation or training examples")
    catalog = OntologyCatalog.load(training_manifest["recipe"]["ontology_version"])
    if catalog.snapshot_sha256 != training_manifest["ontology_snapshot_sha256"]:
        raise ValueError("ontology snapshot differs from the original training run")
    raw = _load(training_predictions_path)
    if not isinstance(raw, list):
        raise ValueError("training predictions must be a list")
    cohort_ids = set(_by_id(examples))
    baseline = [row for row in raw if row["id"] in cohort_ids]
    baseline_metrics = _checked_metrics(
        examples, baseline, training, catalog, _load(training_metrics_path)["validation"])
    comparison = {
        "kind": "paired_validation_comparison_v1", "cohort": "validation",
        "cohort_count": len(examples),
        "source_training_run_id": training_manifest["run_id"],
        "source_training_manifest_sha256": file_sha256(training_manifest_path),
        "selected_epoch": selected_epoch,
        "baseline": {"route": "nf4_adapter", "metrics": _summary(baseline_metrics)},
        "benchmarks": [],
    }
    merged_by_route = {}
    rounds_by_route = {}
    source_model = None
    expected_training = [{"id": row["id"], "gold": row["gold"],
                          "solution": row["solution"]} for row in training]
    for manifest_path, report_path in benchmark_paths:
        raw_manifest = _load(manifest_path)
        manifest, candidate_examples = verify_inference_prepared(
            manifest_path, raw_manifest["code_commit"])
        source = manifest["source_training"]
        if (source["run_id"] != training_manifest["run_id"] or
                source["manifest_sha256"] != comparison["source_training_manifest_sha256"] or
                source["code_commit"] != training_manifest["code_commit"] or
                source["uv_lock_sha256"] != training_manifest["uv_lock_sha256"] or
                source["recipe"] != training_manifest["recipe"] or
                source["ontology_snapshot_sha256"] != catalog.snapshot_sha256 or
                source["selected_epoch"] != selected_epoch):
            raise ValueError("benchmark source training differs from the selected checkpoint")
        if _by_id(candidate_examples) != _by_id(examples):
            raise ValueError("benchmark validation IDs, gold labels or render inputs differ")
        _verify_copied_inputs(training_manifest["files_sha256"],
                              manifest["files_sha256"], candidate_examples)
        if _load(manifest_path.parent / "training_examples.json") != expected_training:
            raise ValueError("benchmark training-frequency reference differs")
        if source_model is None:
            source_model = manifest["source_model"]
            _verify_source_bundle(source_model_bundle, source_model, training_manifest_path,
                                  training_manifest, selected_epoch,
                                  training_predictions_path, training_metrics_path,
                                  baseline_metrics)
        elif source_model != manifest["source_model"]:
            raise ValueError("benchmark source model bundles differ")
        route = manifest["recipe"]["benchmark"].get("route", GGUF_ROUTE)
        if route in merged_by_route:
            raise ValueError("duplicate benchmark route in paired comparison")
        report = _load(report_path)
        expected_kind = ("merged_bf16_inference_benchmark_result_v1" if route == BF16_ROUTE
                         else "inference_benchmark_result_v1")
        expected = {
            "kind": expected_kind, "run_id": manifest["run_id"],
            "source_training": source, "source_model": source_model,
            "benchmark_recipe": manifest["recipe"], "code_commit": manifest["code_commit"],
            "uv_lock_sha256": manifest["uv_lock_sha256"],
            "input_manifest_sha256": file_sha256(manifest_path),
            "cohort": "validation", "cohort_count": len(examples),
        }
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError("benchmark report identity differs from its prepared manifest")
        merged = _merged_hashes(report, route)
        if not isinstance(merged, dict) or not merged:
            raise ValueError("benchmark report lacks merged checkpoint hashes")
        if route == BF16_ROUTE:
            digest = hashlib.sha256(json.dumps(merged, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest()
            if (digest != manifest["recipe"]["benchmark"]["expected_merged_files_digest"] or
                    digest != report["merged_hf"]["files_digest"]):
                raise ValueError("merged BF16 checkpoint hash digest differs")
        merged_by_route[route] = merged
        settings = [item for item in report["settings"] if item["concurrency"] == 1]
        if len(settings) != 1:
            raise ValueError("benchmark requires exactly one concurrency-one setting")
        rounds = settings[0]["rounds"]
        repeats = manifest["recipe"]["benchmark"]["repeats"]
        if (len(rounds) != repeats or
                {item["repeat"] for item in rounds} != set(range(1, repeats + 1))):
            raise ValueError("benchmark has missing or duplicate repeats")
        entry = {"run_id": manifest["run_id"], "route": route,
                 "input_manifest_sha256": expected["input_manifest_sha256"],
                 "report_sha256": file_sha256(report_path), "rounds": []}
        paired_rounds = []
        for round_data in sorted(rounds, key=lambda row: row["repeat"]):
            if round_data["concurrency"] != 1:
                raise ValueError("concurrency-one setting contains another concurrency")
            predictions = round_data["predictions"]
            if _by_id(predictions).keys() != cohort_ids:
                raise ValueError("benchmark prediction IDs differ from validation")
            if any(row.get("image_sha256") != _by_id(examples)[row["id"]]["image_sha256"]
                   for row in predictions):
                raise ValueError("benchmark prediction image hashes differ")
            metrics = _checked_metrics(examples, predictions, training, catalog,
                                       round_data["metrics"])
            entry["rounds"].append({"repeat": round_data["repeat"],
                                    "metrics": _summary(metrics),
                                    "paired_to_nf4": _paired(examples, baseline, predictions, catalog)})
            paired_rounds.append((round_data["repeat"], predictions))
        rounds_by_route[route] = paired_rounds
        comparison["benchmarks"].append(entry)
    if BF16_ROUTE in merged_by_route and GGUF_ROUTE in merged_by_route:
        if merged_by_route[BF16_ROUTE] != merged_by_route[GGUF_ROUTE]:
            raise ValueError("BF16 and GGUF reports used different merged checkpoints")
        comparison["merged_checkpoint_match"] = True
        comparison["bf16_to_gguf"] = [
            {"bf16_repeat": bf16_repeat, "gguf_repeat": gguf_repeat,
             "paired": _paired(examples, bf16_predictions, gguf_predictions, catalog)}
            for bf16_repeat, bf16_predictions in rounds_by_route[BF16_ROUTE]
            for gguf_repeat, gguf_predictions in rounds_by_route[GGUF_ROUTE]
        ]
    return comparison
