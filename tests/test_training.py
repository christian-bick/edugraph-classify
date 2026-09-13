from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from edugraph_classify import training
from edugraph_classify.training import (
    DatasetUploadSpec,
    SupervisedFineTuningSpec,
    TrainingConfigError,
    launch_specs,
    load_experiment_config,
    prepare_run,
)


def test_upload_and_job_specs_validate_provider_resource_identity(tmp_path: Path) -> None:
    artifact = DatasetUploadSpec(
        account_id="account-1",
        dataset_id="dataset-1",
        display_name="Dataset",
        split="train",
        path=tmp_path / "train.jsonl",
        example_count=3,
        sha256="a" * 64,
    )
    job = SupervisedFineTuningSpec(
        account_id="account-1",
        job_id="job-1",
        display_name="Job",
        base_model="accounts/fireworks/models/qwen3-vl-8b-instruct",
        output_model_id="model-1",
        training_dataset=artifact.resource_name,
        evaluation_dataset="accounts/account-1/datasets/validation-1",
        epochs=1,
        lora_rank=8,
        early_stop=False,
        eval_auto_carveout=False,
    )

    assert artifact.resource_name == "accounts/account-1/datasets/dataset-1"
    assert job.output_model_name == "accounts/account-1/models/model-1"

    with pytest.raises(TrainingConfigError, match="resource ID"):
        DatasetUploadSpec(
            account_id="bad/account",
            dataset_id="dataset",
            display_name="Dataset",
            split="train",
            path=tmp_path / "train.jsonl",
            example_count=3,
            sha256="a" * 64,
        )

    invalid_uploads = (
        (lambda: replace(artifact, split="test"), "split"),
        (lambda: replace(artifact, example_count=2), "at least 3"),
        (lambda: replace(artifact, sha256="bad"), "sha256"),
    )
    for build_invalid, message in invalid_uploads:
        with pytest.raises(TrainingConfigError, match=message):
            build_invalid()

    invalid_jobs = (
        (lambda: replace(job, base_model="qwen"), "full Fireworks"),
        (lambda: replace(job, epochs=0), "positive"),
        (lambda: replace(job, lora_rank=2), "lora_rank"),
    )
    for build_invalid, message in invalid_jobs:
        with pytest.raises(TrainingConfigError, match=message):
            build_invalid()


def test_launch_specs_are_derived_from_the_hashed_manifest(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    validation.write_text("{}\n", encoding="utf-8")
    manifest = {
        "run_id": "run-1",
        "provider": {
            "account_id": "account-1",
            "base_model": "accounts/fireworks/models/qwen3-vl-8b-instruct",
            "training_dataset_id": "train-1",
            "evaluation_dataset_id": "validation-1",
            "job_id": "job-1",
            "output_model_id": "model-1",
        },
        "training": {
            "epochs": 1,
            "lora_rank": 8,
            "early_stop": False,
            "eval_auto_carveout": False,
        },
        "dataset": {
            "conversion": {
                "splits": [
                    {
                        "split": "train",
                        "jsonl_path": str(train),
                        "jsonl_sha256": "a" * 64,
                        "example_count": 3,
                    },
                    {
                        "split": "validation",
                        "jsonl_path": str(validation),
                        "jsonl_sha256": "b" * 64,
                        "example_count": 3,
                    },
                ]
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    uploads, job = launch_specs(path)

    assert [item.dataset_id for item in uploads] == ["train-1", "validation-1"]
    assert job.training_dataset == "accounts/account-1/datasets/train-1"
    assert job.evaluation_dataset == "accounts/account-1/datasets/validation-1"


def test_experiment_config_requires_exact_top_level_contract(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"run_id":"run-1"}', encoding="utf-8")
    with pytest.raises(TrainingConfigError, match="exactly"):
        load_experiment_config(path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TrainingConfigError, match="must be an object"):
        load_experiment_config(path)

    payload = {
        "run_id": 1,
        "dataset": {},
        "ontology": {},
        "prompt": {},
        "schema": {},
        "provider": {},
        "training": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingConfigError, match="run_id must be a string"):
        load_experiment_config(path)


def test_tracked_path_validation_rejects_invalid_and_missing_paths(tmp_path: Path) -> None:
    with pytest.raises(TrainingConfigError, match="relative path"):
        training._resolve_tracked_path(tmp_path, None, "prompt.path")
    with pytest.raises(TrainingConfigError, match="does not exist"):
        training._resolve_tracked_path(tmp_path, "missing.json", "prompt.path")


def image_bytes(color: int) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (2, 2), (color, 0, 0)).save(stream, format="PNG")
    return stream.getvalue()


def test_prepare_run_pins_every_identity_and_hashes_generated_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    prompt_path = repo_root / "prompt.json"
    prompt_path.write_text(
        json.dumps(
            {
                "prompt_id": "direct-label",
                "version": "1",
                "system": "system",
                "user": "user",
            }
        ),
        encoding="utf-8",
    )
    schema_path = repo_root / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    revision = "b" * 40
    config = {
        "run_id": "run-1",
        "dataset": {
            "repository": "owner/data",
            "release_tag": "v0.26.0-01",
            "revision": revision,
            "splits": ["train", "validation"],
        },
        "ontology": {
            "package": "edugraph-py",
            "version": "0.26.0",
            "release_asset_sha256": "c" * 64,
        },
        "prompt": {
            "path": "prompt.json",
            "prompt_id": "direct-label",
            "version": "1",
        },
        "schema": {"path": "schema.json", "schema_id": "labels", "version": "1"},
        "model": {"repository": "Qwen/Qwen3.5-4B", "revision": "d" * 40},
        "provider": {
            "account_id": "account-1",
            "base_model": "accounts/fireworks/models/qwen3-vl-8b-instruct",
            "training_dataset_id": "train-1",
            "evaluation_dataset_id": "validation-1",
            "job_id": "job-1",
            "output_model_id": "model-1",
        },
        "training": {
            "epochs": 1,
            "lora_rank": 8,
            "early_stop": False,
            "eval_auto_carveout": False,
        },
    }
    config_path = repo_root / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def rows(split: str, offset: int) -> list[dict[str, object]]:
        return [
            {
                "image": {
                    "path": f"hf://datasets/owner/data@{revision}/{split}/{index}.png",
                    "bytes": image_bytes(offset + index),
                },
                "tags": ["Addition", "DegreeScale", "ProcedureUnderstanding"],
                "solution": bool(index % 2),
            }
            for index in range(3)
        ]

    monkeypatch.setattr(
        training,
        "load_released_splits",
        lambda *args, **kwargs: {
            "train": rows("train", 0),
            "validation": rows("validation", 10),
        },
    )

    prepared = prepare_run(
        config_path,
        repo_root=repo_root,
        runs_root=tmp_path / "runs",
        code_commit="a" * 40,
        workers=2,
    )

    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert prepared.to_mapping()["run_id"] == "run-1"
    assert manifest["code_commit"] == "a" * 40
    assert manifest["dataset"]["conversion"]["splits"][0]["example_count"] == 3
    assert len(manifest["ontology"]["normalized_snapshot_sha256"]) == 64
    assert manifest["ontology"]["normalized_snapshot_format"] == "descriptor-semantics-v2"
    assert len(manifest["prompt"]["sha256"]) == 64
    assert len(manifest["schema"]["sha256"]) == 64
    assert manifest["model"]["revision"] == "d" * 40


def test_prepare_run_rejects_uncommitted_identity_and_escaping_paths(tmp_path: Path) -> None:
    with pytest.raises(TrainingConfigError, match="full lowercase Git commit"):
        prepare_run(
            tmp_path / "missing.json",
            repo_root=tmp_path,
            runs_root=tmp_path / "runs",
            code_commit="short",
        )

    config = {
        "run_id": "run-1",
        "dataset": {"repository": "owner/data", "revision": "b" * 40},
        "ontology": {"version": "0.26.0"},
        "prompt": {"path": "../outside.json", "prompt_id": "x", "version": "1"},
        "schema": {"path": "schema.json"},
        "provider": {},
        "training": {},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(TrainingConfigError, match="escapes"):
        prepare_run(
            config_path,
            repo_root=tmp_path,
            runs_root=tmp_path / "runs",
            code_commit="a" * 40,
        )


def test_prepare_run_validates_smoke_selection_contract(
    tmp_path: Path, monkeypatch
) -> None:
    config = {
        "run_id": "run-1",
        "dataset": {
            "repository": "owner/data",
            "revision": "b" * 40,
            "selection": {"mode": "random", "limits": {"train": 3, "validation": 3}},
        },
        "ontology": {"version": "0.26.0"},
        "prompt": {"path": "prompt.json", "prompt_id": "x", "version": "1"},
        "schema": {"path": "schema.json"},
        "provider": {},
        "training": {},
    }
    (tmp_path / "prompt.json").write_text(
        '{"prompt_id":"x","version":"1","system":"s","user":"u"}',
        encoding="utf-8",
    )
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        training,
        "load_released_splits",
        lambda *args, **kwargs: {"train": [], "validation": []},
    )

    with pytest.raises(TrainingConfigError, match="sorted_prefix"):
        prepare_run(
            path,
            repo_root=tmp_path,
            runs_root=tmp_path / "runs",
            code_commit="a" * 40,
        )
