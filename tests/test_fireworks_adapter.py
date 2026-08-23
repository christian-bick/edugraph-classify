from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from edugraph_classify.contracts import ExecutionMode
from edugraph_classify.preflight import EligibilityStatus
from edugraph_classify.providers import TrainingProvider
from edugraph_classify.providers.fireworks import (
    FireworksTrainingProvider,
    ModelResolutionError,
    TrainingLaunchError,
)
from edugraph_classify.training import (
    DatasetUploadSpec,
    SupervisedFineTuningSpec,
    file_sha256,
)


def model(
    name: str,
    display_name: str,
    *,
    tunable: bool | None = True,
    supervised_lora_tunable: bool | None = True,
) -> SimpleNamespace:
    extra = {}
    if supervised_lora_tunable is not None:
        extra["supervisedLoraTunable"] = supervised_lora_tunable
        extra["useTrainingV2"] = True
    return SimpleNamespace(
        name=name,
        display_name=display_name,
        supports_image_input=True,
        tunable=tunable,
        base_model_details=SimpleNamespace(tunable=False),
        supports_lora=True,
        supports_serverless=False,
        context_length=262_144,
        training_context_length=131_072,
        model_extra=extra,
    )


class FakeModels:
    def __init__(self, models: list[SimpleNamespace]) -> None:
        self.items = models
        self.list_calls: list[tuple[str, int]] = []
        self.get_calls: list[tuple[str, str]] = []

    def list(self, *, account_id: str, page_size: int) -> list[SimpleNamespace]:
        self.list_calls.append((account_id, page_size))
        return self.items

    def get(self, model_id: str, *, account_id: str) -> SimpleNamespace:
        self.get_calls.append((model_id, account_id))
        return next(item for item in self.items if item.name.endswith(f"/{model_id}"))


def provider(*models: SimpleNamespace) -> tuple[FireworksTrainingProvider, FakeModels]:
    resource = FakeModels(list(models))
    return FireworksTrainingProvider(SimpleNamespace(models=resource)), resource


def test_search_resolves_provider_point_notation_without_hard_coding_model_id() -> None:
    adapter, resource = provider(
        model("accounts/fireworks/models/qwen3p5-9b", "Qwen3.5 9B"),
        model("accounts/fireworks/models/qwen3p5-27b", "Qwen3.5 27B"),
        SimpleNamespace(name=None, display_name="ignored"),
    )

    matches = adapter.search("Qwen3.5-9B")

    assert [match.model_id for match in matches] == ["accounts/fireworks/models/qwen3p5-9b"]
    assert resource.list_calls == [("fireworks", 200)]


def test_preflight_maps_sdk_fields_without_leaking_sdk_types() -> None:
    adapter, resource = provider(model("accounts/fireworks/models/qwen3p5-9b", "Qwen3.5 9B"))

    result = adapter.preflight_model("Qwen3.5-9B")

    assert isinstance(adapter, TrainingProvider)
    assert adapter.training_execution_mode is ExecutionMode.PROVIDER_DEDICATED
    assert result.training_execution_mode is ExecutionMode.PROVIDER_DEDICATED
    assert result.status is EligibilityStatus.ELIGIBLE
    assert result.capabilities.identity.provider == "fireworks"
    assert result.capabilities.base_model_tunable is False
    assert result.capabilities.uses_training_v2 is True
    assert resource.get_calls == [("qwen3p5-9b", "fireworks")]


def test_preflight_reports_missing_catalog_tunable_as_ineligible() -> None:
    adapter, _ = provider(
        model("accounts/fireworks/models/qwen3p5-9b", "Qwen3.5 9B", tunable=None)
    )

    result = adapter.preflight_model("Qwen3.5-9B")

    assert result.status is EligibilityStatus.INELIGIBLE
    assert result.reasons == ("provider catalog does not report Tunable: true",)


def test_preflight_rejects_missing_and_ambiguous_hypotheses() -> None:
    adapter, _ = provider()
    with pytest.raises(ModelResolutionError, match="no Fireworks model"):
        adapter.preflight_model("missing")

    adapter, _ = provider(
        model("accounts/fireworks/models/acme-9b-one", "Acme 9B"),
        model("accounts/fireworks/models/acme-9b-two", "Acme 9B"),
    )
    with pytest.raises(ModelResolutionError, match="ambiguous"):
        adapter.preflight_model("Acme 9B")


class FakeDatasets:
    def __init__(self, existing: list[SimpleNamespace] | None = None) -> None:
        self.items = existing or []
        self.calls: list[tuple[str, object]] = []

    def list(self, **kwargs: object) -> list[SimpleNamespace]:
        return self.items

    def create(self, **kwargs: object) -> None:
        self.calls.append(("create", kwargs))

    def upload(self, dataset_id: str, **kwargs: object) -> None:
        self.calls.append(("upload", dataset_id))

    def validate_upload(self, dataset_id: str, **kwargs: object) -> None:
        self.calls.append(("validate", dataset_id))

    def get(self, dataset_id: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            name=f"accounts/account-1/datasets/{dataset_id}",
            state="READY",
            example_count="3",
            status=SimpleNamespace(code="OK", message=""),
        )


class FakeJobs:
    def __init__(self) -> None:
        self.create_kwargs: dict[str, object] | None = None

    def list(self, **kwargs: object) -> list[SimpleNamespace]:
        return []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.create_kwargs = kwargs
        return SimpleNamespace(
            name="accounts/account-1/supervisedFineTuningJobs/job-1",
            state="JOB_STATE_CREATING",
            base_model=kwargs["base_model"],
            output_model="accounts/account-1/models/model-1",
            dataset=kwargs["dataset"],
            evaluation_dataset=kwargs["evaluation_dataset"],
            epochs=kwargs["epochs"],
            lora_rank=kwargs["lora_rank"],
            learning_rate=0.0001,
            max_context_length=32768,
            estimated_cost="3.0",
            status=SimpleNamespace(code="OK", message=""),
        )


def training_provider(
    tmp_path: Path,
    *,
    existing: list[SimpleNamespace] | None = None,
) -> tuple[FireworksTrainingProvider, DatasetUploadSpec, SupervisedFineTuningSpec, FakeDatasets, FakeJobs]:
    path = tmp_path / "train.jsonl"
    path.write_text("{}\n{}\n{}\n", encoding="utf-8")
    datasets = FakeDatasets(existing)
    jobs = FakeJobs()
    models = FakeModels([])
    adapter = FireworksTrainingProvider(
        SimpleNamespace(models=models, datasets=datasets, supervised_fine_tuning_jobs=jobs)
    )
    upload = DatasetUploadSpec(
        account_id="account-1",
        dataset_id="dataset-1",
        display_name="Dataset",
        split="train",
        path=path,
        example_count=3,
        sha256=file_sha256(path),
    )
    job = SupervisedFineTuningSpec(
        account_id="account-1",
        job_id="job-1",
        display_name="Job",
        base_model="accounts/fireworks/models/qwen3-vl-8b-instruct",
        output_model_id="model-1",
        training_dataset=upload.resource_name,
        evaluation_dataset="accounts/account-1/datasets/validation-1",
        epochs=1,
        lora_rank=8,
        early_stop=False,
        eval_auto_carveout=False,
    )
    return adapter, upload, job, datasets, jobs


def test_training_boundary_validates_uploads_and_maps_launch_records(tmp_path: Path) -> None:
    adapter, upload, job, datasets, jobs = training_provider(tmp_path)

    adapter.validate_launch_targets((upload,), job)
    observed = adapter.upload_dataset(upload, timeout_seconds=0, poll_interval_seconds=0)
    launched = adapter.launch_supervised_fine_tuning(job)

    assert observed.name == upload.resource_name
    assert [call[0] for call in datasets.calls] == ["create", "upload", "validate"]
    assert jobs.create_kwargs is not None
    assert jobs.create_kwargs["eval_auto_carveout"] is False
    assert launched.output_model == "accounts/account-1/models/model-1"
    assert launched.estimated_cost == 3.0


def test_training_boundary_refuses_existing_resources(tmp_path: Path) -> None:
    existing = [SimpleNamespace(name="accounts/account-1/datasets/dataset-1")]
    adapter, upload, job, _, _ = training_provider(tmp_path, existing=existing)

    with pytest.raises(TrainingLaunchError, match="refusing to overwrite"):
        adapter.validate_launch_targets((upload,), job)


def test_training_boundary_rechecks_account_file_and_hash(tmp_path: Path) -> None:
    adapter, upload, job, _, _ = training_provider(tmp_path)

    with pytest.raises(TrainingLaunchError, match="same Fireworks account"):
        adapter.validate_launch_targets((replace(upload, account_id="account-2"),), job)
    with pytest.raises(TrainingLaunchError, match="missing"):
        adapter.validate_launch_targets((replace(upload, path=tmp_path / "missing.jsonl"),), job)
    with pytest.raises(TrainingLaunchError, match="hash changed"):
        adapter.validate_launch_targets((replace(upload, sha256="b" * 64),), job)


def test_dataset_upload_reports_provider_rejection_and_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    adapter, upload, _, datasets, _ = training_provider(tmp_path)
    datasets.get = lambda *args, **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        state="FAILED",
        status=SimpleNamespace(code="INVALID_ARGUMENT", message="bad record"),
    )
    with pytest.raises(TrainingLaunchError, match="bad record"):
        adapter.upload_dataset(upload, timeout_seconds=0, poll_interval_seconds=0)

    datasets.get = lambda *args, **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        state="PROCESSING",
        status=SimpleNamespace(code="OK", message=""),
    )
    monkeypatch.setattr("edugraph_classify.providers.fireworks.time.monotonic", lambda: 1.0)
    with pytest.raises(TrainingLaunchError, match="did not become ready"):
        adapter.upload_dataset(upload, timeout_seconds=0, poll_interval_seconds=0)
