"""Thin Fireworks model-catalog adapter."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from typing import Protocol

from ..contracts import ExecutionMode, ModelIdentity
from ..preflight import ModelCapabilities, ModelPreflight, PreflightError, assess_model
from ..training import (
    DatasetUploadSpec,
    ProviderDatasetRecord,
    SupervisedFineTuningSpec,
    TrainingJobRecord,
    file_sha256,
)


class ModelResource(Protocol):
    def list(self, *, account_id: str, page_size: int) -> Iterable[object]: ...

    def get(self, model_id: str, *, account_id: str) -> object: ...


class FireworksClient(Protocol):
    models: ModelResource
    datasets: object
    supervised_fine_tuning_jobs: object


class ModelResolutionError(PreflightError):
    """Raised when a model hypothesis cannot be resolved unambiguously."""


class TrainingLaunchError(RuntimeError):
    """Raised when a guarded Fireworks upload or launch cannot proceed safely."""


def _normalized_model_name(value: str) -> str:
    point_normalized = re.sub(r"(?<=\d)p(?=\d)", ".", value.lower())
    return re.sub(r"[^a-z0-9]", "", point_normalized)


def _model_value(model: object, attribute: str, alias: str) -> object | None:
    value = getattr(model, attribute, None)
    if value is not None:
        return value
    extra = getattr(model, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(alias)
    return None


def _optional_bool(value: object | None) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_int(value: object | None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class FireworksTrainingProvider:
    """Guarded Fireworks capability, dataset, and managed-training adapter."""

    provider_id = "fireworks"
    training_execution_mode = ExecutionMode.PROVIDER_DEDICATED

    def __init__(self, client: FireworksClient, account_id: str = "fireworks") -> None:
        self._client = client
        self._account_id = account_id

    def search(self, hypothesis: str) -> tuple[ModelIdentity, ...]:
        target = _normalized_model_name(hypothesis)
        exact: list[ModelIdentity] = []
        partial: list[ModelIdentity] = []
        for model in self._client.models.list(account_id=self._account_id, page_size=200):
            model_id = getattr(model, "name", None)
            if not isinstance(model_id, str) or not model_id:
                continue
            display_name = getattr(model, "display_name", None)
            names = (model_id.rsplit("/", 1)[-1], display_name if isinstance(display_name, str) else "")
            normalized = tuple(_normalized_model_name(name) for name in names)
            identity = ModelIdentity(provider="fireworks", model_id=model_id)
            if target in normalized:
                exact.append(identity)
            elif any(target in name for name in normalized):
                partial.append(identity)
        matches = exact or partial
        return tuple(sorted(matches, key=lambda item: item.model_id))

    def inspect(self, identity: ModelIdentity) -> ModelCapabilities:
        model_id = identity.model_id.rsplit("/", 1)[-1]
        model = self._client.models.get(model_id, account_id=self._account_id)
        details = getattr(model, "base_model_details", None)
        base_model_tunable = _optional_bool(getattr(details, "tunable", None))
        catalog_tunable = _optional_bool(getattr(model, "tunable", None))
        if catalog_tunable is None:
            # SDK 1.2 omits the computed top-level field returned by older catalog
            # surfaces; baseModelDetails.tunable is the source value firectl exposes.
            catalog_tunable = base_model_tunable
        return ModelCapabilities(
            identity=identity,
            display_name=getattr(model, "display_name", None),
            supports_image_input=_optional_bool(getattr(model, "supports_image_input", None)),
            catalog_tunable=catalog_tunable,
            base_model_tunable=base_model_tunable,
            supports_lora=_optional_bool(getattr(model, "supports_lora", None)),
            supervised_lora_tunable=_optional_bool(
                _model_value(model, "supervised_lora_tunable", "supervisedLoraTunable")
            ),
            uses_training_v2=_optional_bool(_model_value(model, "use_training_v2", "useTrainingV2")),
            supports_serverless_inference=_optional_bool(
                getattr(model, "supports_serverless", None)
            ),
            context_length=_optional_int(getattr(model, "context_length", None)),
            training_context_length=_optional_int(getattr(model, "training_context_length", None)),
        )

    def preflight_model(self, hypothesis: str) -> ModelPreflight:
        matches = self.search(hypothesis)
        if not matches:
            raise ModelResolutionError(f"no Fireworks model matches {hypothesis!r}")
        if len(matches) > 1:
            model_ids = ", ".join(match.model_id for match in matches)
            raise ModelResolutionError(f"Fireworks model hypothesis {hypothesis!r} is ambiguous: {model_ids}")
        return assess_model(self.inspect(matches[0]), self.training_execution_mode)

    def validate_launch_targets(
        self,
        uploads: tuple[DatasetUploadSpec, ...],
        job: SupervisedFineTuningSpec,
        *,
        allowed_existing: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        if any(upload.account_id != job.account_id for upload in uploads):
            raise TrainingLaunchError("all launch resources must use the same Fireworks account")
        for upload in uploads:
            if not upload.path.is_file():
                raise TrainingLaunchError(f"prepared dataset is missing: {upload.path}")
            if file_sha256(upload.path) != upload.sha256:
                raise TrainingLaunchError(f"prepared dataset hash changed: {upload.path}")

        existing = {
            getattr(item, "name", None)
            for item in self._client.datasets.list(account_id=job.account_id, page_size=200)
        }
        existing.update(
            getattr(item, "name", None)
            for item in self._client.supervised_fine_tuning_jobs.list(
                account_id=job.account_id, page_size=200
            )
        )
        existing.update(
            getattr(item, "name", None)
            for item in self._client.models.list(account_id=job.account_id, page_size=200)
        )
        targets = {
            *(upload.resource_name for upload in uploads),
            f"accounts/{job.account_id}/supervisedFineTuningJobs/{job.job_id}",
            job.output_model_name,
        }
        unexpected_allowed = allowed_existing - targets
        if unexpected_allowed:
            raise TrainingLaunchError("resume record contains an unexpected provider resource")
        collisions = sorted(
            target
            for target in targets
            if target in existing and target not in allowed_existing
        )
        if collisions:
            raise TrainingLaunchError(
                f"refusing to overwrite existing Fireworks resources: {', '.join(collisions)}"
            )
        return frozenset(targets & existing)

    def upload_dataset(
        self,
        spec: DatasetUploadSpec,
        *,
        create: bool = True,
        timeout_seconds: float = 300,
        poll_interval_seconds: float = 5,
    ) -> ProviderDatasetRecord:
        if create:
            self._client.datasets.create(
                account_id=spec.account_id,
                dataset_id=spec.dataset_id,
                dataset={
                    "display_name": spec.display_name,
                    "example_count": str(spec.example_count),
                    "format": "CHAT",
                    "user_uploaded": {},
                },
            )
        else:
            existing = self._client.datasets.get(
                spec.dataset_id, account_id=spec.account_id
            )
            if (
                getattr(existing, "name", None) != spec.resource_name
                or getattr(existing, "display_name", None) != spec.display_name
                or getattr(existing, "example_count", None) != str(spec.example_count)
            ):
                raise TrainingLaunchError(
                    f"existing Fireworks dataset does not match resume record: {spec.dataset_id}"
                )
            if getattr(existing, "state", None) == "READY":
                return ProviderDatasetRecord(
                    name=spec.resource_name,
                    state="READY",
                    example_count=spec.example_count,
                )
        self._client.datasets.upload(
            spec.dataset_id,
            account_id=spec.account_id,
            file=(spec.path.name, spec.path, "application/jsonl"),
        )
        self._client.datasets.validate_upload(
            spec.dataset_id,
            account_id=spec.account_id,
            body={},
        )

        deadline = time.monotonic() + timeout_seconds
        while True:
            observed = self._client.datasets.get(spec.dataset_id, account_id=spec.account_id)
            state = getattr(observed, "state", None)
            status = getattr(observed, "status", None)
            status_code = getattr(status, "code", None)
            if state == "READY":
                count = getattr(observed, "example_count", None)
                return ProviderDatasetRecord(
                    name=getattr(observed, "name", spec.resource_name),
                    state=state,
                    example_count=int(count) if isinstance(count, str) and count.isdigit() else None,
                )
            if status_code not in {None, "OK"}:
                message = getattr(status, "message", "dataset validation failed")
                raise TrainingLaunchError(f"Fireworks rejected {spec.dataset_id}: {message}")
            if time.monotonic() >= deadline:
                raise TrainingLaunchError(
                    f"Fireworks dataset {spec.dataset_id} did not become ready in time"
                )
            time.sleep(poll_interval_seconds)

    def launch_supervised_fine_tuning(
        self,
        spec: SupervisedFineTuningSpec,
    ) -> TrainingJobRecord:
        job = self._client.supervised_fine_tuning_jobs.create(
            account_id=spec.account_id,
            supervised_fine_tuning_job_id=spec.job_id,
            dataset=spec.training_dataset,
            evaluation_dataset=spec.evaluation_dataset,
            base_model=spec.base_model,
            output_model=spec.output_model_id,
            display_name=spec.display_name,
            epochs=spec.epochs,
            lora_rank=spec.lora_rank,
            early_stop=spec.early_stop,
            eval_auto_carveout=spec.eval_auto_carveout,
        )
        return self._training_job_record(job, spec)

    def get_supervised_fine_tuning_job(
        self,
        spec: SupervisedFineTuningSpec,
    ) -> TrainingJobRecord:
        job = self._client.supervised_fine_tuning_jobs.get(
            spec.job_id, account_id=spec.account_id
        )
        return self._training_job_record(job, spec)

    @staticmethod
    def _training_job_record(
        job: object,
        spec: SupervisedFineTuningSpec,
    ) -> TrainingJobRecord:
        status = getattr(job, "status", None)
        estimated_cost = getattr(job, "estimated_cost", None)
        return TrainingJobRecord(
            name=getattr(
                job,
                "name",
                f"accounts/{spec.account_id}/supervisedFineTuningJobs/{spec.job_id}",
            ),
            state=getattr(job, "state", None),
            base_model=getattr(job, "base_model", None),
            output_model=getattr(job, "output_model", None),
            dataset=getattr(job, "dataset", None),
            evaluation_dataset=getattr(job, "evaluation_dataset", None),
            epochs=getattr(job, "epochs", None),
            lora_rank=getattr(job, "lora_rank", None),
            learning_rate=getattr(job, "learning_rate", None),
            max_context_length=getattr(job, "max_context_length", None),
            estimated_cost=(
                float(estimated_cost) if isinstance(estimated_cost, (float, int, str)) else None
            ),
            status_code=getattr(status, "code", None),
            status_message=getattr(status, "message", None),
        )
