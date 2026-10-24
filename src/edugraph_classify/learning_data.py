"""Pinned, deterministic learning cohorts and portable prompt artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
from importlib.metadata import version
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from PIL import Image

from .dataset import PromptTemplate, file_sha256
from .ontology import OntologyCatalog
from .output_contract import closed_vocabulary_schema
from .qwen_training_policy import audit_qwen_language_targets
from .qwen_rendering import QwenVisionRenderer

RUNTIME_PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "peft", "bitsandbytes",
                    "wandb", "lm-format-enforcer", "edugraph-py", "pillow")


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError("artifact paths must be safe relative paths")
    return value


def load_recipe(path: Path, *, extra_fields: frozenset[str] = frozenset()) -> dict:
    return validate_recipe(json.loads(path.read_text(encoding="utf-8")), extra_fields=extra_fields)


def validate_selection(selection: dict) -> None:
    """Numeric cohorts stay balanced; full splits preserve every released row."""
    train, final = selection["train"], selection.get("final_validation", 0)
    if train != "all" and (type(train) is not int or train <= 0):
        raise ValueError("selection.train must be a positive integer or 'all'")
    if final != "remaining" and (type(final) is not int or final < 0):
        raise ValueError("selection.final_validation must be a nonnegative integer or 'remaining'")
    for name in ("validation", "train_diagnostic"):
        if type(selection[name]) is not int or selection[name] <= 0:
            raise ValueError(f"selection.{name} must be a positive integer")
    if any(value % 2 for value in (train, selection["validation"], selection["train_diagnostic"], final)
           if type(value) is int):
        raise ValueError("balanced question/solution cohorts require even counts")
    if type(train) is int and selection["train_diagnostic"] > train:
        raise ValueError("training diagnostic must be a subset of training")


def validate_recipe(config: dict, *, extra_fields: frozenset[str] = frozenset()) -> dict:
    """Validate an in-memory recipe without filesystem or provider side effects."""
    expected = {"run_id", "dataset", "ontology_version", "prompt_path", "schema_path", "model",
                "selection", "training", "evaluation", "pricing", "estimated_cost_limit_usd"}
    if set(config) != expected | extra_fields or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", config["run_id"]):
        raise ValueError("invalid learning recipe fields or run_id")
    for revision in (config["dataset"]["revision"], config["model"]["hf_revision"]):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("dataset and model must use immutable commit revisions")
    if config["dataset"]["label_field"] != "labels" or config["dataset"]["source_ontology_version"] != config["ontology_version"]:
        raise ValueError("dataset labels contract and ontology must match")
    validate_selection(config["selection"])
    for section, names in (("training", ("epochs", "batch_size", "rank", "alpha", "max_context_tokens")),
                           ("evaluation", ("max_tokens",))):
        for name in names:
            value = config[section][name]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{section}.{name} must be a positive integer")
    if set(config["pricing"]) != {"compute_hour_usd", "disk_hour_usd", "estimated_hours"}:
        raise ValueError("pricing must specify Runpod compute, disk, and estimated hours")
    for value in (*config["pricing"].values(), config["estimated_cost_limit_usd"], config["training"]["learning_rate"]):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError("learning rate, prices, and cost limit must be positive finite numbers")
    if config["evaluation"]["temperature"] != 0:
        raise ValueError("paired learning evaluation requires greedy sampling")
    if config["evaluation"].get("output_mode") != "json_schema":
        raise ValueError("Runpod evaluation requires json_schema output")
    if config["evaluation"].get("every_epoch") is not True:
        raise ValueError("Runpod evaluation must generate predictions every epoch")
    patience = config["training"].get("early_stop_patience")
    if patience is not None and (type(patience) is not int or patience <= 0 or not config["evaluation"].get("every_epoch", False)):
        raise ValueError("early_stop_patience requires every-epoch evaluation and a positive integer")
    targets = config["training"].get("expected_target_modules")
    if targets is not None and (not isinstance(targets, list) or any(not isinstance(v, str) for v in targets)):
        raise ValueError("training.expected_target_modules must be a list of module names")
    if targets is not None and any(name not in config["training"] for name in ("train_attn", "train_mlp", "train_unembed")):
        raise ValueError("explicit LoRA targets require explicit SDK train_attn, train_mlp, and train_unembed flags")
    for name in ("train_attn", "train_mlp", "train_unembed"):
        if name in config["training"] and type(config["training"][name]) is not bool:
            raise ValueError(f"training.{name} must be boolean")
    safe_relative(config["prompt_path"])
    safe_relative(config["schema_path"])
    return config


def metadata_rows(content: bytes, catalog: OntologyCatalog) -> list[dict]:
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    paths = set()
    for row in rows:
        if set(row) != {"file_name", "labels", "solution"} or type(row["solution"]) is not bool:
            raise ValueError("released metadata violates the pinned labels contract")
        safe_relative(row["file_name"])
        if row["file_name"] in paths:
            raise ValueError("duplicate source path in official split")
        paths.add(row["file_name"])
        catalog.labels(row["labels"])
    return rows


def select_cohort(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Hash-sample each public view; filename is an opaque locator, never a group."""
    selected = []
    for solution in (False, True):
        candidates = [row for row in rows if row["solution"] == solution]
        candidates.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['file_name']}".encode()).hexdigest())
        if len(candidates) < count // 2:
            raise ValueError("not enough examples in one question/solution stratum")
        selected.extend(candidates[:count // 2])
    return sorted(selected, key=lambda row: row["file_name"])


def complete_cohort(rows: list[dict], name: str) -> list[dict]:
    """Retain the released view distribution, including odd or unequal counts."""
    if not rows:
        raise ValueError(f"{name} requested a complete cohort but no examples remain")
    return sorted(rows, key=lambda row: row["file_name"])


def label_profile(rows: list[dict], catalog: OntologyCatalog) -> dict:
    """Describe explicit released labels without changing gold or deriving closure."""
    frequencies = Counter(label for row in rows for label in row["labels"])
    return {
        "rows": len(rows),
        "views": {name: sum(row["solution"] == flag for row in rows)
                  for name, flag in (("question", False), ("solution", True))},
        "observed_labels": len(frequencies),
        "label_support": dict(sorted(frequencies.items())),
        "max_labels": max((len(row["labels"]) for row in rows), default=0),
        "max_labels_by_dimension": {
            dimension: max((sum(catalog.dimensions[label] == dimension for label in row["labels"])
                            for row in rows), default=0)
            for dimension in ("areas", "scopes", "abilities")},
    }


def conflicting_duplicate_images(examples: list[dict]) -> list[dict]:
    """Flag identical train images with distinct explicit gold sets without changing either row."""
    groups = {}
    for row in examples:
        if row["split"] == "train":
            groups.setdefault(row["image_sha256"], []).append(row)
    return [{"image_sha256": digest,
             "examples": [{"id": row["id"], "gold": row["gold"]} for row in sorted(rows, key=lambda item: item["id"])]}
            for digest, rows in sorted(groups.items())
            if len({row["target"] for row in rows}) > 1]


def prepare_learning(config: dict, root: Path, run: Path, processor: Path, code_commit: str,
                     fetch, capabilities: dict, renderer_factory=QwenVisionRenderer.load,
                     runtime_packages=RUNTIME_PACKAGES) -> dict:
    """Fetch public sources and prepare locally; never upload or start compute."""
    validate_selection(config["selection"])
    catalog = OntologyCatalog.load(config["ontology_version"])
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "recipe.json", config)
    schema = json.loads((root / config["schema_path"]).read_text(encoding="utf-8"))
    write_json(run / "schema.json", schema)
    write_json(run / "closed_schema.json", closed_vocabulary_schema(schema, catalog))
    prompt = PromptTemplate.load(root / config["prompt_path"]).to_mapping()
    write_json(run / "prompt.json", prompt)
    (run / "processor").mkdir()
    for file in sorted(processor.iterdir()):
        if file.is_file() and not file.name.startswith("."):
            shutil.copyfile(file, run / "processor" / file.name)
    target_audit = None
    if config["training"].get("expected_target_modules") is not None:
        target_audit = audit_qwen_language_targets(run / "processor", config["training"]["expected_target_modules"])
        write_json(run / "language_target_audit.json", target_audit)
    renderer = renderer_factory(run / "processor", prompt)
    (run / "chat_template.jinja").write_text(renderer.export_template(), encoding="utf-8")
    (run / "images").mkdir()
    dataset = config["dataset"]
    final_validation = config["selection"].get("final_validation", 0)
    base = f"https://huggingface.co/datasets/{dataset['repository']}/resolve/{dataset['revision']}"
    counts, selected, official = {}, [], {}
    for split in ("train", "validation"):
        content = fetch(f"{base}/{split}/metadata.jsonl")
        (run / f"{split}-metadata.jsonl").write_bytes(content)
        rows = metadata_rows(content, catalog)
        official[split] = rows
        counts[split] = len(rows)
        requested = config["selection"][split]
        cohort = (complete_cohort(rows, "selection.train") if requested == "all" else
                  select_cohort(rows, requested, config["selection"]["seed"]))
        selected.extend({**row, "split": split} for row in cohort)
        if split == "validation" and final_validation:
            chosen = {row["file_name"] for row in cohort}
            remaining = [row for row in rows if row["file_name"] not in chosen]
            final = (complete_cohort(remaining, "selection.final_validation") if final_validation == "remaining" else
                     select_cohort(remaining, final_validation, config["selection"]["seed"] + 2))
            selected.extend({**row, "split": "final_validation"} for row in final)

    def convert(row):
        source_split = "validation" if row["split"] == "final_validation" else row["split"]
        source = f"{source_split}/{row['file_name']}"
        data = fetch(f"{base}/{source}")
        if len(data) > 7_500_000:
            raise ValueError(f"{source}: source image exceeds the 7.5 MB preparation limit")
        with Image.open(io.BytesIO(data)) as img:
            size = list(img.size)
            img.verify()
        digest = hashlib.sha256(data).hexdigest()
        image_path = f"images/{hashlib.sha256(source.encode()).hexdigest()}.png"
        (run / image_path).write_bytes(data)
        gold = catalog.labels(row["labels"])
        try:
            rendered = renderer.render(data, gold.to_json(), max_tokens=config["training"]["max_context_tokens"], max_output=config["evaluation"]["max_tokens"])
        except ValueError as error:
            raise ValueError(f"{source}: {error}") from error
        return {"id": source, "file_name": row["file_name"], "split": row["split"], "solution": row["solution"],
                "image_path": image_path, "image_sha256": digest, "image_size": size,
                "gold": gold.to_mapping(), "target": gold.to_json(), "render_sha256": rendered.fingerprint(),
                "training_tokens": rendered.training_tokens, "prompt_tokens": rendered.prompt_tokens,
                "image_tokens": rendered.image_tokens, "supervised_tokens": len(rendered.completion)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        examples = list(pool.map(convert, selected))
    train_hashes = {row["image_sha256"] for row in examples if row["split"] == "train"}
    validation_hashes = {row["image_sha256"] for row in examples if row["split"] in ("validation", "final_validation")}
    if train_hashes & validation_hashes:
        raise ValueError("identical image bytes cross selected official splits")
    if ({row["image_sha256"] for row in examples if row["split"] == "validation"} &
            {row["image_sha256"] for row in examples if row["split"] == "final_validation"}):
        raise ValueError("identical image bytes cross selection and final validation cohorts")
    train = [row for row in examples if row["split"] == "train"]
    validation = [row for row in examples if row["split"] == "validation"]
    diagnostic = select_cohort(train, config["selection"]["train_diagnostic"], config["selection"]["seed"] + 1)
    for row in examples:
        row["train_diagnostic"] = row["id"] in {v["id"] for v in diagnostic}
    write_json(run / "examples.json", examples)
    final = [row for row in examples if row["split"] == "final_validation"]
    sampling = validation + diagnostic
    training_tokens = sum(row["training_tokens"] for row in train) * config["training"]["epochs"]
    evaluation_rounds = 1 + (config["training"]["epochs"] if config["evaluation"].get("every_epoch", False) else 1)
    prefill_tokens = evaluation_rounds * sum(row["prompt_tokens"] for row in sampling) + sum(row["prompt_tokens"] for row in final)
    output_tokens = (evaluation_rounds * len(sampling) + len(final)) * config["evaluation"]["max_tokens"]
    price = config["pricing"]
    cost = price["estimated_hours"] * (price["compute_hour_usd"] + price["disk_hour_usd"])
    if cost > config["estimated_cost_limit_usd"]:
        raise ValueError("prepared cost estimate exceeds recipe limit")
    profiles = {split: label_profile(rows, catalog) for split, rows in official.items()}
    cohorts = {split: label_profile([row for row in selected if row["split"] == split], catalog)
               for split in ("train", "validation", "final_validation")}
    resolved = {"train": len(train), "validation": len(validation), "final_validation": len(final),
                "train_diagnostic": len(diagnostic)}
    manifest = {
        "run_id": config["run_id"], "code_commit": code_commit, "recipe": config,
        "runtime_versions": {name: version(name).split("+")[0] for name in runtime_packages},
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
        "ontology_snapshot_sha256": catalog.snapshot_sha256, "capabilities_at_preparation": capabilities,
        "metadata_rows_validated": counts, "selected_images_validated": len(examples),
        "duplicate_image_counts": {split: sum(n - 1 for n in Counter(r["image_sha256"] for r in examples if r["split"] == split).values()) for split in ("train", "validation", "final_validation")},
        "training_data_conflicts": conflicting_duplicate_images(examples),
        "split_policy": "Official train/validation split preserved; byte overlap checked on selected images. Public metadata has no task-group identity; semantic independence is not established.",
        "selection_policy": "Numeric counts use SHA256(seed:opaque_filename) within equally weighted question/solution strata. train='all' retains every official training row; final_validation='remaining' retains every row outside checkpoint selection. Full cohorts preserve released view counts. Final assessment comes only from the official-validation complement and is excluded from checkpoint selection.",
        "cohort_counts": {split: sum(r["split"] == split for r in examples) for split in ("train", "validation", "final_validation")},
        "resolved_selection": resolved,
        "unused_metadata_rows": {"train": counts["train"] - len(train),
                                 "validation": counts["validation"] - len(validation) - len(final)},
        "data_profile": {"eligible_ontology_labels": len(catalog.eligible_labels),
            "official_splits": profiles, "cohorts": cohorts,
            "train_labels_missing_from_official_validation": sorted(set(profiles["train"]["label_support"]) - set(profiles["validation"]["label_support"])),
            "train_labels_missing_from_selected_validation": sorted(set(cohorts["train"]["label_support"]) - set(cohorts["validation"]["label_support"]))},
        "token_limits": {"configured_context_tokens": config["training"]["max_context_tokens"],
            "configured_output_tokens": config["evaluation"]["max_tokens"],
            "max_training_tokens": max(row["training_tokens"] for row in examples),
            "max_prompt_tokens": max(row["prompt_tokens"] for row in examples),
            "max_image_tokens": max(row["image_tokens"] for row in examples),
            "max_completion_tokens": max(row["supervised_tokens"] for row in examples),
            "required_context_tokens": max(max(row["training_tokens"], row["prompt_tokens"] + config["evaluation"]["max_tokens"]) for row in examples)},
        "optimizer_steps": math.ceil(len(train) / config["training"]["batch_size"]) * config["training"]["epochs"],
        "token_budget": {"train": training_tokens, "prefill": prefill_tokens, "sample": output_tokens},
        "estimated_run_cost_usd": cost, "estimate_is_not_provider_billing_cap": True,
        "quality_promotion": False, "prompt_role": "system", "thinking": False,
        "language_target_audit": target_audit,
        "template_export": "Portable Jinja template with the fixed classifier system message. Verify serving-runtime template support separately.",
        "files_sha256": {p.relative_to(run).as_posix(): file_sha256(p) for p in sorted(run.rglob("*")) if p.is_file()},
    }
    write_json(run / "manifest.json", manifest)
    return manifest


def verify_prepared(path: Path, code_commit: str) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["code_commit"] != code_commit:
        raise ValueError("launch requires the same clean code commit as preparation")
    for name, digest in manifest["files_sha256"].items():
        location = path.parent / safe_relative(name)
        if not location.resolve().is_relative_to(path.parent.resolve()) or file_sha256(location) != digest:
            raise ValueError("prepared artifact hash changed or path escaped run directory")
    if manifest["recipe"] != json.loads((path.parent / "recipe.json").read_text(encoding="utf-8")):
        raise ValueError("manifest recipe differs from its hashed prepared copy")
    return manifest, json.loads((path.parent / "examples.json").read_text(encoding="utf-8"))
