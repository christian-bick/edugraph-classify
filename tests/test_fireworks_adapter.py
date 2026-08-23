from __future__ import annotations

from types import SimpleNamespace

import pytest

from edugraph_classify.contracts import ExecutionMode
from edugraph_classify.preflight import EligibilityStatus
from edugraph_classify.providers import TrainingProvider
from edugraph_classify.providers.fireworks import (
    FireworksTrainingProvider,
    ModelResolutionError,
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
