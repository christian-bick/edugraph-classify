"""Provider-neutral experiment preparation and managed-training records."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .dataset import (
    ConversionResult,
    PromptTemplate,
    convert_dataset,
    file_sha256,
    load_released_splits,
    select_sorted_prefix,
)
from .ontology import OntologyCatalog


class TrainingConfigError(ValueError):
    """Raised when a tracked experiment configuration is incomplete or inconsistent."""


_RESOURCE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _require_resource_id(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _RESOURCE_ID.fullmatch(value):
        raise TrainingConfigError(f"{field_name} is not a valid provider resource ID")


def _require_keys(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise TrainingConfigError(f"{name} must contain exactly {sorted(expected)}")


def _mapping(payload: Mapping[str, object], field_name: str) -> Mapping[str, object]:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        raise TrainingConfigError(f"{field_name} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class DatasetUploadSpec:
    account_id: str
    dataset_id: str
    display_name: str
    split: str
    path: Path
    example_count: int
    sha256: str

    def __post_init__(self) -> None:
        _require_resource_id(self.account_id, "account_id")
        _require_resource_id(self.dataset_id, "dataset_id")
        if self.split not in {"train", "validation"}:
            raise TrainingConfigError("dataset split must be train or validation")
        if self.example_count < 3:
            raise TrainingConfigError("Fireworks datasets require at least 3 examples")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise TrainingConfigError("dataset sha256 is invalid")

    @property
    def resource_name(self) -> str:
        return f"accounts/{self.account_id}/datasets/{self.dataset_id}"


@dataclass(frozen=True, slots=True)
class ProviderDatasetRecord:
    name: str
    state: str | None
    example_count: int | None


@dataclass(frozen=True, slots=True)
class SupervisedFineTuningSpec:
    account_id: str
    job_id: str
    display_name: str
    base_model: str
    output_model_id: str
    training_dataset: str
    evaluation_dataset: str
    epochs: int
    lora_rank: int
    early_stop: bool
    eval_auto_carveout: bool

    def __post_init__(self) -> None:
        _require_resource_id(self.account_id, "account_id")
        _require_resource_id(self.job_id, "job_id")
        _require_resource_id(self.output_model_id, "output_model_id")
        if not self.base_model.startswith("accounts/fireworks/models/"):
            raise TrainingConfigError("base_model must be a full Fireworks catalog name")
        if self.epochs < 1:
            raise TrainingConfigError("epochs must be positive")
        if self.lora_rank not in {4, 8, 16, 32}:
            raise TrainingConfigError("lora_rank must be one of 4, 8, 16, or 32")

    @property
    def output_model_name(self) -> str:
        return f"accounts/{self.account_id}/models/{self.output_model_id}"


@dataclass(frozen=True, slots=True)
class TrainingJobRecord:
    name: str
    state: str | None
    base_model: str | None
    output_model: str | None
    dataset: str | None
    evaluation_dataset: str | None
    epochs: int | None
    lora_rank: int | None
    learning_rate: float | None
    max_context_length: int | None
    estimated_cost: float | None
    status_code: str | None
    status_message: str | None

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreparedRun:
    run_id: str
    manifest_path: Path
    manifest_sha256: str
    conversion: ConversionResult

    def to_mapping(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "conversion": self.conversion.to_mapping(),
        }


def load_experiment_config(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TrainingConfigError("experiment configuration must be an object")
    _require_keys(
        payload,
        {"run_id", "dataset", "ontology", "prompt", "schema", "provider", "training"},
        "experiment configuration",
    )
    run_id = payload["run_id"]
    if not isinstance(run_id, str):
        raise TrainingConfigError("run_id must be a string")
    _require_resource_id(run_id, "run_id")
    return payload


def _resolve_tracked_path(repo_root: Path, value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TrainingConfigError(f"{field_name} must be a relative path")
    path = (repo_root / value).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise TrainingConfigError(f"{field_name} escapes the repository") from error
    if not path.is_file():
        raise TrainingConfigError(f"{field_name} does not exist")
    return path


def _sha256_json(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def prepare_run(
    config_path: Path,
    *,
    repo_root: Path,
    runs_root: Path,
    code_commit: str,
    workers: int = 8,
) -> PreparedRun:
    if not _COMMIT.fullmatch(code_commit):
        raise TrainingConfigError("code_commit must be a full lowercase Git commit")
    config = load_experiment_config(config_path)
    dataset_config = _mapping(config, "dataset")
    ontology_config = _mapping(config, "ontology")
    prompt_config = _mapping(config, "prompt")
    schema_config = _mapping(config, "schema")
    provider_config = _mapping(config, "provider")
    training_config = _mapping(config, "training")

    repo_id = dataset_config.get("repository")
    revision = dataset_config.get("revision")
    ontology_version = ontology_config.get("version")
    if not isinstance(repo_id, str) or not isinstance(revision, str):
        raise TrainingConfigError("dataset repository and revision must be strings")
    if not isinstance(ontology_version, str):
        raise TrainingConfigError("ontology version must be a string")

    prompt_path = _resolve_tracked_path(repo_root, prompt_config.get("path"), "prompt.path")
    schema_path = _resolve_tracked_path(repo_root, schema_config.get("path"), "schema.path")
    prompt = PromptTemplate.load(prompt_path)
    if prompt.prompt_id != prompt_config.get("prompt_id") or prompt.version != prompt_config.get(
        "version"
    ):
        raise TrainingConfigError("prompt identity does not match its tracked file")
    catalog = OntologyCatalog.load(ontology_version)

    run_id = config["run_id"]
    assert isinstance(run_id, str)
    run_dir = runs_root / run_id
    data_dir = run_dir / "data"
    splits = load_released_splits(repo_id, revision, cache_dir=run_dir / "cache")
    selection = dataset_config.get("selection")
    if selection is not None:
        if not isinstance(selection, Mapping):
            raise TrainingConfigError("dataset.selection must be an object")
        _require_keys(selection, {"mode", "limits"}, "dataset.selection")
        if selection["mode"] != "sorted_prefix":
            raise TrainingConfigError(
                "dataset.selection mode must be sorted_prefix when provided"
            )
        limits = selection["limits"]
        if not isinstance(limits, Mapping):
            raise TrainingConfigError("dataset.selection.limits must be an object")
        splits = select_sorted_prefix(splits, limits=limits)
    conversion = convert_dataset(
        splits,
        repo_id=repo_id,
        revision=revision,
        catalog=catalog,
        prompt=prompt,
        output_dir=data_dir,
        workers=workers,
    )

    manifest = {
        "manifest_version": 1,
        "run_id": run_id,
        "code_commit": code_commit,
        "experiment_config": {
            "path": str(config_path),
            "sha256": _sha256_json(config_path),
        },
        "dataset": {**dataset_config, "conversion": conversion.to_mapping()},
        "ontology": {
            **ontology_config,
            "normalized_snapshot_sha256": catalog.snapshot_sha256,
        },
        "prompt": {**prompt_config, "sha256": prompt.sha256},
        "schema": {**schema_config, "sha256": _sha256_json(schema_path)},
        "provider": dict(provider_config),
        "training": dict(training_config),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return PreparedRun(
        run_id=run_id,
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        conversion=conversion,
    )


def launch_specs(manifest_path: Path) -> tuple[tuple[DatasetUploadSpec, ...], SupervisedFineTuningSpec]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provider = manifest["provider"]
    training = manifest["training"]
    split_records = {
        record["split"]: record for record in manifest["dataset"]["conversion"]["splits"]
    }
    account_id = provider["account_id"]
    uploads = tuple(
        DatasetUploadSpec(
            account_id=account_id,
            dataset_id=provider[
                "training_dataset_id" if split == "train" else "evaluation_dataset_id"
            ],
            display_name=f"{manifest['run_id']} {split}",
            split=split,
            path=Path(split_records[split]["jsonl_path"]),
            example_count=split_records[split]["example_count"],
            sha256=split_records[split]["jsonl_sha256"],
        )
        for split in ("train", "validation")
    )
    job = SupervisedFineTuningSpec(
        account_id=account_id,
        job_id=provider["job_id"],
        display_name=manifest["run_id"],
        base_model=provider["base_model"],
        output_model_id=provider["output_model_id"],
        training_dataset=uploads[0].resource_name,
        evaluation_dataset=uploads[1].resource_name,
        epochs=training["epochs"],
        lora_rank=training["lora_rank"],
        early_stop=training["early_stop"],
        eval_auto_carveout=training["eval_auto_carveout"],
    )
    return uploads, job
