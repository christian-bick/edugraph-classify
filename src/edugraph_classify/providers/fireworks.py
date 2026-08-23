"""Thin Fireworks model-catalog adapter."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

from ..contracts import ExecutionMode, ModelIdentity
from ..preflight import ModelCapabilities, ModelPreflight, PreflightError, assess_model


class ModelResource(Protocol):
    def list(self, *, account_id: str, page_size: int) -> Iterable[object]: ...

    def get(self, model_id: str, *, account_id: str) -> object: ...


class FireworksClient(Protocol):
    models: ModelResource


class ModelResolutionError(PreflightError):
    """Raised when a model hypothesis cannot be resolved unambiguously."""


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
    """Read-only Fireworks training capability adapter."""

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
