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

from .dataset import file_sha256
from .ontology import OntologyCatalog
from .output_contract import closed_vocabulary_schema
from .qwen_training_policy import audit_qwen_language_targets
from .qwen_rendering import QwenVisionRenderer


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError("artifact paths must be safe relative paths")
    return value


def load_recipe(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {"run_id", "dataset", "ontology_version", "prompt_path", "schema_path", "model",
                "selection", "training", "evaluation", "pricing", "estimated_cost_limit_usd"}
    if set(config) != expected or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", config["run_id"]):
        raise ValueError("invalid learning recipe fields or run_id")
    for revision in (config["dataset"]["revision"], config["model"]["hf_revision"]):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("dataset and model must use immutable commit revisions")
    if config["dataset"]["label_field"] != "labels" or config["dataset"]["source_ontology_version"] != config["ontology_version"]:
        raise ValueError("dataset labels contract and ontology must match")
    for section, names in (("selection", ("train", "validation", "train_diagnostic")),
                           ("training", ("epochs", "batch_size", "rank", "alpha", "max_context_tokens")),
                           ("evaluation", ("max_tokens",))):
        for name in names:
            value = config[section][name]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{section}.{name} must be a positive integer")
    if any(config["selection"][k] % 2 for k in ("train", "validation", "train_diagnostic")):
        raise ValueError("balanced question/solution cohorts require even counts")
    if config["selection"]["train_diagnostic"] > config["selection"]["train"]:
        raise ValueError("training diagnostic must be a subset of training")
    for value in (*config["pricing"].values(), config["estimated_cost_limit_usd"], config["training"]["learning_rate"]):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError("learning rate, prices, and cost limit must be positive finite numbers")
    if config["evaluation"]["temperature"] != 0:
        raise ValueError("paired learning evaluation requires greedy sampling")
    if config["evaluation"].get("output_mode", "raw") not in ("raw", "json_schema"):
        raise ValueError("evaluation.output_mode must be raw or json_schema")
    if "every_epoch" in config["evaluation"] and type(config["evaluation"]["every_epoch"]) is not bool:
        raise ValueError("evaluation.every_epoch must be boolean")
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


def system_prompt(source: dict, catalog: OntologyCatalog) -> dict:
    vocabulary = {field: sorted(name for name in catalog.eligible_labels if catalog.dimensions[name] == field)
                  for field in ("areas", "scopes", "abilities")}
    return {**source, "system": source["system"] + "\n\nAllowed ontology vocabulary (v" + catalog.package_version + "):\n"
            + json.dumps(vocabulary, ensure_ascii=False, separators=(",", ":"))}


def prepare_learning(config: dict, root: Path, run: Path, processor: Path, code_commit: str,
                     fetch, capabilities: dict, renderer_factory=QwenVisionRenderer.load) -> dict:
    """Fetch public sources and prepare locally; never upload or start compute."""
    catalog = OntologyCatalog.load(config["ontology_version"])
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "recipe.json", config)
    schema = json.loads((root / config["schema_path"]).read_text(encoding="utf-8"))
    write_json(run / "schema.json", schema)
    write_json(run / "closed_schema.json", closed_vocabulary_schema(schema, catalog))
    prompt = system_prompt(json.loads((root / config["prompt_path"]).read_text(encoding="utf-8")), catalog)
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
    base = f"https://huggingface.co/datasets/{dataset['repository']}/resolve/{dataset['revision']}"
    counts, selected = {}, []
    for split in ("train", "validation"):
        content = fetch(f"{base}/{split}/metadata.jsonl")
        (run / f"{split}-metadata.jsonl").write_bytes(content)
        rows = metadata_rows(content, catalog)
        counts[split] = len(rows)
        selected.extend({**row, "split": split} for row in select_cohort(rows, config["selection"][split], config["selection"]["seed"]))

    def convert(row):
        source = f"{row['split']}/{row['file_name']}"
        data = fetch(f"{base}/{source}")
        if len(data) > 7_500_000:
            raise ValueError("source image exceeds the 10 MB base64 payload budget")
        with Image.open(io.BytesIO(data)) as img:
            size = list(img.size)
            img.verify()
        digest = hashlib.sha256(data).hexdigest()
        image_path = f"images/{hashlib.sha256(source.encode()).hexdigest()}.png"
        (run / image_path).write_bytes(data)
        gold = catalog.labels(row["labels"])
        rendered = renderer.render(data, gold.to_json(), max_tokens=config["training"]["max_context_tokens"], max_output=config["evaluation"]["max_tokens"])
        return {"id": source, "file_name": row["file_name"], "split": row["split"], "solution": row["solution"],
                "image_path": image_path, "image_sha256": digest, "image_size": size,
                "gold": gold.to_mapping(), "target": gold.to_json(), "render_sha256": rendered.fingerprint(),
                "training_tokens": rendered.training_tokens, "prompt_tokens": rendered.prompt_tokens,
                "image_tokens": rendered.image_tokens, "supervised_tokens": len(rendered.completion)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        examples = list(pool.map(convert, selected))
    train_hashes = {row["image_sha256"] for row in examples if row["split"] == "train"}
    validation_hashes = {row["image_sha256"] for row in examples if row["split"] == "validation"}
    if train_hashes & validation_hashes:
        raise ValueError("identical image bytes cross selected official splits")
    train = [row for row in examples if row["split"] == "train"]
    validation = [row for row in examples if row["split"] == "validation"]
    diagnostic = select_cohort(train, config["selection"]["train_diagnostic"], config["selection"]["seed"] + 1)
    for row in examples:
        row["train_diagnostic"] = row["id"] in {v["id"] for v in diagnostic}
    write_json(run / "examples.json", examples)
    sampling = validation + diagnostic
    training_tokens = sum(row["training_tokens"] for row in train) * config["training"]["epochs"]
    evaluation_rounds = 1 + (config["training"]["epochs"] if config["evaluation"].get("every_epoch", False) else 1)
    prefill_tokens = evaluation_rounds * sum(row["prompt_tokens"] for row in sampling)
    output_tokens = evaluation_rounds * len(sampling) * config["evaluation"]["max_tokens"]
    price = config["pricing"]
    cost = (training_tokens * price["train"] + prefill_tokens * price["prefill"] + output_tokens * price["sample"]) / 1_000_000
    if cost > config["estimated_cost_limit_usd"]:
        raise ValueError("prepared token-cost upper estimate exceeds recipe limit")
    manifest = {
        "run_id": config["run_id"], "code_commit": code_commit, "recipe": config,
        "runtime_versions": {name: version(name) for name in ("fireworks-ai", "edugraph-py", "pillow")},
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
        "ontology_snapshot_sha256": catalog.snapshot_sha256, "capabilities_at_preparation": capabilities,
        "metadata_rows_validated": counts, "selected_images_validated": len(examples),
        "duplicate_image_counts": {split: sum(n - 1 for n in Counter(r["image_sha256"] for r in examples if r["split"] == split).values()) for split in ("train", "validation")},
        "split_policy": "Official splits preserved; byte overlap checked on selected images. Public metadata has no task-group identity; semantic independence is not established.",
        "selection_policy": "SHA256(seed:opaque_filename) within equally weighted question/solution strata; validation is not a representative-weighted full-release score.",
        "optimizer_steps": math.ceil(len(train) / config["training"]["batch_size"]) * config["training"]["epochs"],
        "token_budget": {"train": training_tokens, "prefill": prefill_tokens, "sample": output_tokens},
        "estimated_max_token_cost_usd": cost, "estimate_is_not_provider_billing_cap": True,
        "quality_promotion": False, "prompt_role": "system", "thinking": False,
        "language_target_audit": target_audit,
        "template_export": "Portable base-model Jinja template; Fireworks LoRA template override is not supported by published docs.",
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
