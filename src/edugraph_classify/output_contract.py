"""Provider-neutral, ontology-pinned JSON output contract."""

from __future__ import annotations

from copy import deepcopy

from .ontology import OntologyCatalog

DIMENSIONS = ("areas", "scopes", "abilities")


def closed_vocabulary_schema(base_schema: dict, catalog: OntologyCatalog) -> dict:
    """Limit each dimension to eligible labels without changing set semantics."""
    if set(base_schema.get("properties", {})) != set(DIMENSIONS):
        raise ValueError("output schema must contain exactly the ontology dimensions")
    schema = deepcopy(base_schema)
    schema["required"] = list(DIMENSIONS)
    schema["additionalProperties"] = False
    for field in DIMENSIONS:
        prop = schema["properties"][field]
        if prop.get("type") != "array" or prop.get("items", {}).get("type") != "string":
            raise ValueError("each ontology dimension must be an array of strings")
        prop["items"]["enum"] = sorted(
            name for name in catalog.eligible_labels if catalog.dimensions[name] == field
        )
        if not prop["items"]["enum"]:
            raise ValueError("ontology dimension has no eligible labels")
    return schema
