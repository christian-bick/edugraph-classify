"""Prepare a hardware benchmark locally from verified smoke evidence; no upload."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from edugraph_classify.dataset import file_sha256
from edugraph_classify.learning_data import select_cohort, verify_prepared, write_json
from edugraph_classify.ontology import OntologyCatalog
from edugraph_classify.qwen_rendering import QwenVisionRenderer
from edugraph_classify.runpod_config import validate_runpod_recipe
from edugraph_classify.self_hosted_run import verify_resume

from runpod_hardware import ordered_batches, verified_spec


def prepare(source, evidence, profile, destination, harness_commit):
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest, examples = verify_prepared(source / "manifest.json", original["code_commit"])
    # The old digest-pinned image remains authoritative for all training code.
    changes = subprocess.check_output(["git", "diff", original["code_commit"], harness_commit,
        "--", "src", "pyproject.toml", "uv.lock"], text=True)
    if changes:
        raise ValueError("benchmark cannot substitute changed trainer code")
    manifest = deepcopy(manifest)
    config = manifest["recipe"]
    config["run_id"] = manifest["run_id"] = destination.name
    config["training"]["epochs"] = 2  # Required to validate a continuation from epoch one.
    config["execution"].update(gpu_type="NVIDIA RTX 6000 Ada Generation", minimum_vram_gib=45,
                               minimum_ram_gib=48, minimum_vcpus=8, max_hours=1, max_compute_hour_usd=1)
    config["pricing"].update(compute_hour_usd=0.84, disk_hour_usd=0.024, estimated_hours=0.5)
    config["estimated_cost_limit_usd"] = 1
    validate_runpod_recipe(config)
    destination.mkdir(parents=True, exist_ok=False)
    for name in manifest["files_sha256"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    write_json(destination / "recipe.json", config)
    manifest["files_sha256"]["recipe.json"] = file_sha256(destination / "recipe.json")
    checkpoint = json.loads((evidence / "checkpoint-verification.json").read_text())
    state = verify_resume(manifest, Path(checkpoint["local_directory"]))
    if (state["epoch"], state["step"], state["phase"]) != (1, 40, "final"):
        raise ValueError("expected the completed epoch-one source checkpoint")
    observed = json.loads((source / "observed-event-summary.json").read_text())
    baseline = next(r["reference"] for r in observed["checkpoints"] if r["step"] == 0)
    model = Path(json.loads((evidence / "model-verification.json").read_text())["local_directory"])
    predictions = {r["id"]: r for r in json.loads((model / "reports/epoch-1-predictions.json").read_text())}
    validation = [r for r in examples if r["split"] == "validation"]
    selected = select_cohort(validation[2:], 20, config["selection"]["seed"])
    seen = {}
    for line in (source / "observed-logs.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["source"] == "container" and item["line"].startswith('{"event": "optimizer_step"'):
            event = json.loads(item["line"])
            seen[event["step"]] = event
    batches = ordered_batches(examples, config["training"], 0, 20)
    a40_training = [{**seen[i], "ids": [r["id"] for r in rows]} for i, rows in enumerate(batches, 1)]
    catalog = OntologyCatalog.load(config["ontology_version"])
    metadata = {r["file_name"]: r for r in map(json.loads, (source / "train-metadata.jsonl").read_text().splitlines())}
    all_rows = json.loads(profile.read_text())
    longest = [max((r for r in all_rows if r["split"] == "train" and r["solution"] == solution),
                   key=lambda r: r["training_tokens"]) for solution in (False, True)]
    renderer = QwenVisionRenderer.load(destination / "processor", json.loads((source / "prompt.json").read_text()))
    memory_rows = []
    for row in longest:
        data = Path(row["source"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
            raise ValueError("longest-example source bytes changed")
        relative = "benchmark-images/" + row["image_sha256"] + ".png"
        target = destination / relative
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
        gold = catalog.labels(metadata[row["file_name"]]["labels"])
        rendered = renderer.render(data, gold.to_json(), max_tokens=config["training"]["max_context_tokens"],
                                   max_output=config["evaluation"]["max_tokens"])
        if rendered.fingerprint() != row["render_sha256"]:
            raise ValueError("longest-example rendering differs from full-dataset audit")
        memory_rows.append({**{k: v for k, v in row.items() if k != "source"}, "image_path": relative,
                            "gold": gold.to_mapping(), "target": gold.to_json(), "train_diagnostic": False})
    spec = {"training_steps": 20, "warmup_training_steps": 2, "generation_ids": [r["id"] for r in selected],
            "a40_training": a40_training, "a40_generation": [predictions[r["id"]] for r in selected],
            "memory_examples": memory_rows,
            "baseline_checkpoint": {k: baseline[k] for k in ("uri", "sha256")},
            "final_checkpoint": {k: checkpoint["reference"][k] for k in ("uri", "sha256")}}
    write_json(destination / "benchmark.json", spec)
    write_json(destination / "resume-reference.json", checkpoint["reference"])
    manifest["hardware_benchmark"] = {"harness_commit": harness_commit,
        "harness_sha256": file_sha256(Path(__file__).with_name("runpod_hardware.py")),
        "spec_path": "benchmark.json", "spec_sha256": file_sha256(destination / "benchmark.json"),
        "source_run_id": original["run_id"], "mode": "bounded_hardware_diagnostic",
        "planned_optimizer_updates": 22, "comparison_steps": 20, "comparison_generations": 20,
        "live_resume_updates": 1, "discarded_memory_probe_updates": 1}
    train_rows = [r for batch in batches for r in batch] + ordered_batches(examples, config["training"], 1, 1)[0] + memory_rows * 2
    generation_rows = selected[:2] + selected + [max(memory_rows, key=lambda r: r["prompt_tokens"])]
    manifest.update(optimizer_steps=22, estimated_run_cost_usd=0.5 * 0.864,
        token_budget={"train": sum(r["training_tokens"] for r in train_rows),
                      "prefill": sum(r["prompt_tokens"] for r in generation_rows),
                      "sample": len(generation_rows) * config["evaluation"]["max_tokens"]})
    write_json(destination / "manifest.json", manifest)
    verify_prepared(destination / "manifest.json", original["code_commit"])
    verified_spec(destination, manifest)
    verify_resume(manifest, Path(checkpoint["local_directory"]))
    return {"manifest": str(destination / "manifest.json"), "run_id": manifest["run_id"],
            "estimated_cost_usd": manifest["estimated_run_cost_usd"], "harness": manifest["hardware_benchmark"],
            "memory_probe_training_tokens": [r["training_tokens"] for r in memory_rows]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--harness-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.evidence, args.profile, args.destination, args.harness_commit), indent=2))
