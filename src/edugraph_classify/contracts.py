"""Provider-neutral identities and prediction contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ContractError(ValueError):
    """Raised when data violates a provider-neutral classifier contract."""


class ExecutionMode(StrEnum):
    """Infrastructure ownership, intentionally independent from provider identity."""

    SERVERLESS = "serverless"
    PROVIDER_DEDICATED = "provider_dedicated"
    SELF_HOSTED = "self_hosted"


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ContractError(f"{field_name} must not contain surrounding whitespace")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Immutable upstream or generated artifact reference."""

    repository: str
    revision: str
    content_hash: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.repository, "repository")
        _require_text(self.revision, "revision")
        if self.content_hash is not None:
            _require_text(self.content_hash, "content_hash")


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Exact model identity without provider SDK objects."""

    provider: str
    model_id: str
    revision: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.provider, "provider")
        _require_text(self.model_id, "model_id")
        if self.revision is not None:
            _require_text(self.revision, "revision")


@dataclass(frozen=True, slots=True)
class PromptSchemaIdentity:
    """Versioned prompt and output-schema pair."""

    prompt_id: str
    prompt_version: str
    schema_id: str
    schema_version: str

    def __post_init__(self) -> None:
        for field_name in ("prompt_id", "prompt_version", "schema_id", "schema_version"):
            _require_text(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Minimum reproducibility identity shared by all execution adapters."""

    run_id: str
    dataset: ArtifactIdentity
    ontology: ArtifactIdentity
    prompt_schema: PromptSchemaIdentity
    code_commit: str
    model: ModelIdentity
    execution_mode: ExecutionMode

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        _require_text(self.code_commit, "code_commit")


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True, slots=True)
class ClassificationRequest:
    """Identity for one atomic classification unit in a pinned run."""

    request_id: str
    run_id: str
    source_id: str
    image_sha256: str
    solution: bool

    def __post_init__(self) -> None:
        _require_text(self.request_id, "request_id")
        _require_text(self.run_id, "run_id")
        _require_text(self.source_id, "source_id")
        if not _SHA256_RE.fullmatch(self.image_sha256):
            raise ContractError("image_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "image_sha256", self.image_sha256.lower())
        if not isinstance(self.solution, bool):
            raise ContractError("solution must be a boolean")


def _label_tuple(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractError(f"{field_name} must be an array of strings")
    labels = tuple(values)
    for label in labels:
        _require_text(label, f"{field_name} label")
    return labels


@dataclass(frozen=True, slots=True)
class LabelSet:
    """Dimension-aware explicit or derived labels with set semantics."""

    areas: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    abilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("areas", "scopes", "abilities"):
            object.__setattr__(self, field_name, _label_tuple(getattr(self, field_name), field_name))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> LabelSet:
        expected = {"areas", "scopes", "abilities"}
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise ContractError(f"prediction fields must be exactly {sorted(expected)}; missing={missing}, extra={extra}")
        return cls(
            areas=_label_tuple(payload["areas"], "areas"),
            scopes=_label_tuple(payload["scopes"], "scopes"),
            abilities=_label_tuple(payload["abilities"], "abilities"),
        )

    def to_mapping(self) -> dict[str, list[str]]:
        return {
            "areas": list(self.areas),
            "scopes": list(self.scopes),
            "abilities": list(self.abilities),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class CanonicalizationResult:
    """Deterministic structural cleanup and its visible findings."""

    labels: LabelSet
    findings: tuple[str, ...] = ()


def canonicalize_structure(labels: LabelSet) -> CanonicalizationResult:
    """Remove exact duplicates and sort identifiers without ontology inference."""

    canonical: dict[str, tuple[str, ...]] = {}
    findings: list[str] = []
    for field_name in ("areas", "scopes", "abilities"):
        values = getattr(labels, field_name)
        ordered = tuple(sorted(set(values)))
        if len(ordered) != len(values):
            findings.append(f"removed duplicate {field_name} labels")
        if ordered != values:
            findings.append(f"sorted {field_name} labels")
        canonical[field_name] = ordered
    return CanonicalizationResult(labels=LabelSet(**canonical), findings=tuple(findings))


@dataclass(frozen=True, slots=True)
class RawPrediction:
    """Provider response retained before parsing or canonicalization."""

    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise ContractError("raw prediction content must be a string")


@dataclass(frozen=True, slots=True)
class PredictionViews:
    """Non-overwriting progression from provider output to ontology derivation."""

    raw: RawPrediction
    parsed: LabelSet | None = None
    canonical: LabelSet | None = None
    derived: LabelSet | None = None
    findings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.canonical is not None and self.parsed is None:
            raise ContractError("canonical labels require parsed labels")
        if self.derived is not None and self.canonical is None:
            raise ContractError("derived labels require canonical labels")
