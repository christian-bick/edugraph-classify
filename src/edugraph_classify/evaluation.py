"""Deterministic explicit-label evaluation; no repair or ontology closure."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter

from .contracts import ContractError, LabelSet, canonicalize_structure
from .ontology import OntologyCatalog

DIMENSIONS = ("areas", "scopes", "abilities")


def parse_prediction(text: str, catalog: OntologyCatalog) -> dict:
    """Keep raw validity separate from sorted/deduplicated structural output."""
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict) or any(not isinstance(v, list) for v in payload.values()):
            raise ContractError("prediction must be a dimension object with arrays")
        parsed = LabelSet.from_mapping(payload)
    except (ValueError, TypeError):
        return {"valid": False, "canonical_valid": False, "canonical": None, "findings": ["invalid_json_or_schema"]}
    canonical = canonicalize_structure(parsed)
    invalid = [name for field in DIMENSIONS for name in getattr(parsed, field)
               if name not in catalog.eligible_labels or catalog.dimensions.get(name) != field]
    duplicates = any(len(set(getattr(parsed, f))) != len(getattr(parsed, f)) for f in DIMENSIONS)
    return {
        "valid": not invalid and not duplicates,
        "canonical_valid": not invalid,
        "canonical": canonical.labels.to_mapping(),
        "findings": [*canonical.findings, *[f"invalid_label:{v}" for v in invalid]],
    }


def label_pairs(labels: dict) -> set[tuple[str, str]]:
    return {(field, label) for field in DIMENSIONS for label in labels[field]}


def counts_metrics(tp: int, fp: int, fn: int) -> dict:
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def _metrics(rows: list[dict], dimension: str | None = None, allowed: set | None = None) -> dict:
    tp = fp = fn = exact = 0
    per_example = []
    for row in rows:
        gold, prediction = row["gold_pairs"], row["pred_pairs"]
        if dimension:
            gold = {v for v in gold if v[0] == dimension}
            prediction = {v for v in prediction if v[0] == dimension}
        if allowed is not None:
            gold, prediction = gold & allowed, prediction & allowed
        a, b, c = len(gold & prediction), len(prediction - gold), len(gold - prediction)
        tp += a
        fp += b
        fn += c
        exact += row["valid"] and gold == prediction
        per_example.append(counts_metrics(a, b, c)["f1"])
    return {"n": len(rows), "exact_set_match": exact / len(rows) if rows else None,
            "example_mean_f1": statistics.mean(per_example) if rows else None,
            **counts_metrics(tp, fp, fn)}


def evaluate(examples: list[dict], predictions: list[dict], training: list[dict], catalog: OntologyCatalog) -> dict:
    expected = [row["id"] for row in examples]
    actual = [row["id"] for row in predictions]
    if len(set(expected)) != len(expected) or len(set(actual)) != len(actual) or set(expected) != set(actual):
        raise ValueError("evaluation requires exactly one prediction for each selected example")
    by_id = {row["id"]: row for row in predictions}
    rows, canonical_rows, assessments = [], [], []
    for example in examples:
        result = by_id[example["id"]]
        parsed = parse_prediction(result["text"], catalog)
        pairs = label_pairs(parsed["canonical"]) if parsed["canonical"] else set()
        # Unknown/misdimensioned labels remain false positives; unparseable JSON
        # supplies no usable labels and misses every gold label.
        row = {"valid": parsed["valid"], "gold_pairs": label_pairs(example["gold"]),
               "pred_pairs": pairs, "solution": example["solution"]}
        rows.append(row)
        canonical_rows.append({**row, "valid": parsed["canonical_valid"]})
        assessments.append({"id": example["id"], **parsed})
    frequencies = Counter(pair for example in training for pair in label_pairs(example["gold"]))
    vocabulary = {(catalog.dimensions[name], name) for name in catalog.eligible_labels}
    slices = {"unseen": {v for v in vocabulary if frequencies[v] == 0},
              "rare_1_4": {v for v in vocabulary if 1 <= frequencies[v] <= 4},
              "common_5_plus": {v for v in vocabulary if frequencies[v] >= 5}}
    latencies = sorted(float(p["latency_seconds"]) for p in predictions)
    label_metrics = [_metrics(rows, allowed={v}) for v in vocabulary
                     if any(v in row["gold_pairs"] or v in row["pred_pairs"] for row in rows)]
    return {
        "explicit": _metrics(rows), "canonical": _metrics(canonical_rows),
        "label_macro_f1": statistics.mean(v["f1"] for v in label_metrics) if label_metrics else 0.0,
        "invalid_output_rate": sum(not r["valid"] for r in rows) / len(rows) if rows else None,
        "per_dimension": {field: _metrics(rows, field) for field in DIMENSIONS},
        "per_view": {name: _metrics([r for r in rows if r["solution"] == flag])
                     for name, flag in (("question", False), ("solution", True))},
        "training_frequency_slices": {name: _metrics(rows, allowed=labels) for name, labels in slices.items()},
        "latency_seconds": {"median": statistics.median(latencies) if latencies else None,
                            "p95": latencies[math.ceil(len(latencies) * .95) - 1] if latencies else None},
        "usage": {key: sum(p.get("usage", {}).get(key, 0) for p in predictions)
                  for key in ("prompt_tokens", "completion_tokens")},
        "assessments": assessments,
    }
