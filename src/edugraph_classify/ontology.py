"""Pinned EduGraph ontology catalog used by conversion and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from types import MappingProxyType
from typing import ClassVar, Mapping

from edugraph import Ability, Area, Scope, relations

from .contracts import LabelSet


class OntologyError(ValueError):
    """Raised when the pinned ontology cannot resolve a dataset label safely."""


@dataclass(frozen=True, slots=True)
class OntologyCatalog:
    snapshot_format: ClassVar[str] = "descriptor-semantics-v2"
    package_version: str
    dimensions: Mapping[str, str]
    iris: Mapping[str, str]
    eligible_labels: frozenset[str]
    snapshot_sha256: str

    @classmethod
    def load(cls, expected_version: str) -> OntologyCatalog:
        package_version = version("edugraph-py")
        if package_version != expected_version:
            raise OntologyError(
                f"expected edugraph-py {expected_version}, found {package_version}"
            )

        dimensions: dict[str, str] = {}
        iris: dict[str, str] = {}
        eligible_labels: set[str] = set()
        snapshot: list[dict[str, object]] = []
        for field_name, enum_type in (
            ("areas", Area),
            ("scopes", Scope),
            ("abilities", Ability),
        ):
            for member in enum_type:
                if member.name in dimensions:
                    raise OntologyError(
                        f"ontology label {member.name!r} occurs in multiple dimensions"
                    )
                dimensions[member.name] = field_name
                iris[member.name] = member.value
                descriptor_relations = relations(member)
                # ONT-E7: constituent children make a node organizational.
                # Specialization children do not prevent direct labeling.
                eligible = not descriptor_relations.get("hasPart")
                if eligible:
                    eligible_labels.add(member.name)
                snapshot.append(
                    {
                        "dimension": field_name,
                        "local_name": member.name,
                        "iri": member.value,
                        "definition": member.definition,
                        "relations": {
                            name: sorted({target.value for target in targets})
                            for name, targets in descriptor_relations.items()
                            if name != "definition"
                        },
                        "eligible_for_direct_labeling": eligible,
                    }
                )

        serialized = json.dumps(
            {
                "format": cls.snapshot_format,
                "descriptors": sorted(
                    snapshot, key=lambda item: (item["dimension"], item["local_name"])
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            package_version=package_version,
            dimensions=MappingProxyType(dimensions),
            iris=MappingProxyType(iris),
            eligible_labels=frozenset(eligible_labels),
            snapshot_sha256=hashlib.sha256(serialized).hexdigest(),
        )

    def labels(self, tags: object) -> LabelSet:
        if not isinstance(tags, list):
            raise OntologyError("dataset tags must be an array of strings")

        grouped: dict[str, list[str]] = {
            "areas": [],
            "scopes": [],
            "abilities": [],
        }
        seen: set[str] = set()
        for tag in tags:
            if not isinstance(tag, str) or not tag:
                raise OntologyError("dataset tags must contain non-empty strings")
            if tag in seen:
                raise OntologyError(f"dataset row repeats ontology label {tag!r}")
            seen.add(tag)
            dimension = self.dimensions.get(tag)
            if dimension is None:
                raise OntologyError(f"unknown ontology label {tag!r}")
            if tag not in self.eligible_labels:
                raise OntologyError(
                    f"ontology label {tag!r} is organizational (hasPart children); "
                    "not eligible for direct labeling"
                )
            grouped[dimension].append(tag)

        return LabelSet(
            areas=tuple(sorted(grouped["areas"])),
            scopes=tuple(sorted(grouped["scopes"])),
            abilities=tuple(sorted(grouped["abilities"])),
        )
