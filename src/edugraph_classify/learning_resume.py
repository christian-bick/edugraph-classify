"""Offline provenance checks for continuing a saved training state in a new run."""

from __future__ import annotations

import json
from pathlib import Path

from .dataset import file_sha256
from .learning_data import conflicting_duplicate_images, verify_prepared


def _training_policy(recipe: dict) -> dict:
    return {key: value for key, value in recipe["training"].items()
            if key not in ("epochs", "early_stop_patience")}


def _selection_policy(recipe: dict) -> dict:
    # Final validation is never used for optimizer steps or checkpoint selection.
    return {key: value for key, value in recipe["selection"].items() if key != "final_validation"}


def _learning_rows(examples: list[dict]) -> list[dict]:
    return sorted((row for row in examples if row["split"] in ("train", "validation")),
                  key=lambda row: row["id"])


def verify_resume_source(target_manifest: dict, target_examples: list[dict],
                         source_execution_path: Path, source_epoch: int) -> dict:
    """Verify a completed run's exact learning cohort and saved optimizer state."""
    if type(source_epoch) is not int or source_epoch <= 0:
        raise ValueError("source_epoch must be a positive integer")
    source_manifest_path = source_execution_path.parent / "manifest.json"
    pinned_commit = json.loads(source_manifest_path.read_text(encoding="utf-8"))["code_commit"]
    source_manifest, source_examples = verify_prepared(source_manifest_path, pinned_commit)
    execution = json.loads(source_execution_path.read_text(encoding="utf-8"))
    if execution.get("status") != "completed" or execution.get("run_id") != source_manifest["run_id"]:
        raise ValueError("resume source must be a completed, matching training run")
    if execution.get("manifest_sha256") != file_sha256(source_manifest_path):
        raise ValueError("source execution no longer matches its prepared manifest")
    if source_manifest["run_id"] == target_manifest["run_id"]:
        raise ValueError("continuation requires a new run_id and prepared directory")
    if conflicting_duplicate_images(source_examples) or conflicting_duplicate_images(target_examples):
        raise ValueError("conflicting identical training images must be resolved before continuation")
    source, target = source_manifest["recipe"], target_manifest["recipe"]
    if not target["evaluation"].get("every_epoch", False):
        raise ValueError("continuation requires every-epoch checkpoint evaluation")
    stable_fields = ("dataset", "ontology_version", "prompt_path", "schema_path", "model", "evaluation")
    if any(source[field] != target[field] for field in stable_fields):
        raise ValueError("resume model, dataset, ontology, prompt, schema, and evaluation policy must match")
    if _training_policy(source) != _training_policy(target) or _selection_policy(source) != _selection_policy(target):
        raise ValueError("resume training and checkpoint-selection policy must match")
    if any(source_manifest[key] != target_manifest[key] for key in
           ("ontology_snapshot_sha256", "runtime_versions", "uv_lock_sha256", "language_target_audit")):
        raise ValueError("resume runtime or language-only model layout differs")
    source_files, target_files = source_manifest["files_sha256"], target_manifest["files_sha256"]
    identity_files = {"prompt.json", "schema.json", "closed_schema.json", "chat_template.jinja"}
    source_processor = {name: digest for name, digest in source_files.items() if name.startswith("processor/")}
    target_processor = {name: digest for name, digest in target_files.items() if name.startswith("processor/")}
    if source_processor != target_processor:
        raise ValueError("resume processor bytes differ")
    if any(source_files.get(name) != target_files.get(name) for name in identity_files):
        raise ValueError("resume prompt, schema, or processor bytes differ")
    if _learning_rows(source_examples) != _learning_rows(target_examples):
        raise ValueError("resume training or checkpoint-selection examples differ")
    states = [event["path"] for event in execution.get("events", [])
              if event.get("stage") == "training_state_saved" and event.get("epoch") == source_epoch]
    provider_run = execution.get("provider", {}).get("run_id")
    if len(states) != 1 or not isinstance(states[0], str) or not isinstance(provider_run, str) or \
            f"/{provider_run}/" not in states[0]:
        raise ValueError("source epoch has no unique fully qualified training-state checkpoint")
    return {"source_run_id": source_manifest["run_id"], "source_epoch": source_epoch,
            "source_code_commit": source_manifest["code_commit"],
            "source_manifest_sha256": file_sha256(source_manifest_path),
            "source_execution_sha256": file_sha256(source_execution_path),
            "training_state_path": states[0]}
