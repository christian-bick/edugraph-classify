"""Provider-neutral managed-training boundary."""

from typing import Mapping, Protocol, runtime_checkable

from ..contracts import ExecutionMode
from ..preflight import ModelPreflight
from ..training import (
    DatasetUploadSpec,
    ProviderDatasetRecord,
    SupervisedFineTuningSpec,
    TrainingJobRecord,
)


@runtime_checkable
class TrainingProvider(Protocol):
    """Read-only capabilities required before any paid training workflow."""

    provider_id: str
    training_execution_mode: ExecutionMode

    def preflight_model(self, hypothesis: str) -> ModelPreflight: ...

    def validate_launch_targets(
        self,
        uploads: tuple[DatasetUploadSpec, ...],
        job: SupervisedFineTuningSpec,
        *,
        allowed_existing: frozenset[str] = frozenset(),
    ) -> frozenset[str]: ...

    def upload_dataset(
        self, spec: DatasetUploadSpec, *, create: bool = True
    ) -> ProviderDatasetRecord: ...

    def launch_supervised_fine_tuning(
        self,
        spec: SupervisedFineTuningSpec,
    ) -> TrainingJobRecord: ...

    def get_supervised_fine_tuning_job(
        self,
        spec: SupervisedFineTuningSpec,
    ) -> TrainingJobRecord: ...


@runtime_checkable
class ServerlessTrainingProvider(Protocol):
    """Boundary for provider-managed custom-container training jobs."""

    provider_id: str
    training_execution_mode: ExecutionMode

    def render_custom_job(self, spec: object) -> Mapping[str, object]: ...

    def find_custom_jobs(self, spec: object) -> tuple[object, ...]: ...

    def launch_custom_job(self, spec: object) -> object: ...
