from edugraph_classify.contracts import ExecutionMode, ModelIdentity
from edugraph_classify.preflight import (
    EligibilityStatus,
    ModelCapabilities,
    assess_model,
)


def capabilities(**changes: object) -> ModelCapabilities:
    values = {
        "identity": ModelIdentity("fireworks", "accounts/fireworks/models/model"),
        "display_name": "Model",
        "supports_image_input": True,
        "catalog_tunable": True,
        "base_model_tunable": True,
        "supports_lora": True,
        "supervised_lora_tunable": True,
        "uses_training_v2": True,
        "supports_serverless_inference": False,
        "context_length": 262_144,
        "training_context_length": 131_072,
    }
    values.update(changes)
    return ModelCapabilities(**values)  # type: ignore[arg-type]


def test_model_is_eligible_only_when_all_required_capabilities_are_explicit() -> None:
    result = assess_model(
        capabilities(catalog_tunable=False, supervised_lora_tunable=True, uses_training_v2=True),
        ExecutionMode.SERVERLESS,
    )

    assert result.status is EligibilityStatus.ELIGIBLE
    assert result.reasons == ()
    assert result.to_mapping()["status"] == "eligible"
    assert result.to_mapping()["training_execution_mode"] == "serverless"


def test_v2_routing_alone_and_missing_lora_do_not_grant_eligibility():
    result = assess_model(capabilities(supervised_lora_tunable=None, supports_lora=None), ExecutionMode.SERVERLESS)
    assert result.status is EligibilityStatus.INELIGIBLE
    assert len(result.reasons) == 2


def test_model_capability_check_fails_closed() -> None:
    result = assess_model(
        capabilities(
            supports_image_input=None,
            catalog_tunable=None,
            supervised_lora_tunable=None,
            uses_training_v2=None,
        ),
        ExecutionMode.PROVIDER_DEDICATED,
    )

    assert result.status is EligibilityStatus.INELIGIBLE
    assert result.reasons == (
        "model does not report vision input support",
        "provider catalog does not report Tunable: true",
        "model does not report a compatible supervised training surface",
    )
