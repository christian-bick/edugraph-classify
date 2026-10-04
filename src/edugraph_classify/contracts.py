"""Dimension-aware label sets and deterministic structural canonicalization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class ContractError(ValueError):
    """Raised when labels violate the classifier output contract."""


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ContractError(f"{field_name} must not contain surrounding whitespace")
    return value


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
