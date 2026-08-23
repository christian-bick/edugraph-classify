"""Provider-neutral managed-training boundary."""

from typing import Protocol, runtime_checkable

from ..contracts import ExecutionMode
from ..preflight import ModelPreflight


@runtime_checkable
class TrainingProvider(Protocol):
    """Read-only capabilities required before any paid training workflow."""

    provider_id: str
    training_execution_mode: ExecutionMode

    def preflight_model(self, hypothesis: str) -> ModelPreflight: ...
