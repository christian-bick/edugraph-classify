"""Provider-neutral model capability assessment."""

from dataclasses import asdict, dataclass
from enum import StrEnum

from .contracts import ExecutionMode, ModelIdentity


class PreflightError(RuntimeError):
    """Base class for safe, user-facing preflight failures."""


class EligibilityStatus(StrEnum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    identity: ModelIdentity
    display_name: str | None
    supports_image_input: bool | None
    catalog_tunable: bool | None
    base_model_tunable: bool | None
    supports_lora: bool | None
    supervised_lora_tunable: bool | None
    uses_training_v2: bool | None
    supports_serverless_inference: bool | None
    context_length: int | None
    training_context_length: int | None


@dataclass(frozen=True, slots=True)
class ModelPreflight:
    training_execution_mode: ExecutionMode
    capabilities: ModelCapabilities
    status: EligibilityStatus
    reasons: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload["training_execution_mode"] = self.training_execution_mode.value
        payload["status"] = self.status.value
        payload["capabilities"]["identity"] = asdict(self.capabilities.identity)
        return payload


def assess_model(
    capabilities: ModelCapabilities,
    training_execution_mode: ExecutionMode,
) -> ModelPreflight:
    """Apply the accepted fail-closed vision SFT eligibility policy."""

    reasons: list[str] = []
    if capabilities.supports_image_input is not True:
        reasons.append("model does not report vision input support")
    # Fireworks documents `tunable` as legacy V1; V2 advertises separate
    # supervised/RL flags. Neither V2 routing nor inference hosting is eligibility.
    if capabilities.uses_training_v2 is not True and capabilities.catalog_tunable is not True:
        reasons.append("provider catalog does not report Tunable: true")
    if capabilities.supervised_lora_tunable is not True:
        reasons.append("model does not report a compatible supervised training surface")
    if capabilities.supports_lora is not True:
        reasons.append("model does not report LoRA support")

    status = EligibilityStatus.INELIGIBLE if reasons else EligibilityStatus.ELIGIBLE
    return ModelPreflight(
        training_execution_mode=training_execution_mode,
        capabilities=capabilities,
        status=status,
        reasons=tuple(reasons),
    )
