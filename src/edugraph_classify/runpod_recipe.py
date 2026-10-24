"""Create a release-specific recipe without changing the validated training policy."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

from .learning_data import validate_recipe
from .runpod_config import validate_runpod_recipe


def new_release_recipe(template: dict, *, run_id: str, dataset_release: str,
                       dataset_revision: str, ontology_version: str,
                       estimated_hours: float, max_hours: float, cost_limit_usd: float,
                       context_tokens: int | None = None) -> dict:
    """Retain model/optimizer/provider pins; require a fresh identity and budget."""
    if run_id == template["run_id"]:
        raise ValueError("a new release requires a new run ID")
    if not dataset_release or not ontology_version:
        raise ValueError("dataset release and ontology version must be explicit")
    config = deepcopy(template)
    config["run_id"] = run_id
    config["ontology_version"] = ontology_version
    config["dataset"].update(release=dataset_release, revision=dataset_revision,
                             source_ontology_version=ontology_version)
    config["selection"].update(train="all", final_validation="remaining")
    if context_tokens is not None:
        config["training"]["max_context_tokens"] = context_tokens
    config["pricing"]["estimated_hours"] = estimated_hours
    config["execution"]["max_hours"] = max_hours
    config["estimated_cost_limit_usd"] = cost_limit_usd
    validate_recipe(config, extra_fields=frozenset({"execution", "tracking", "artifacts"}))
    return validate_runpod_recipe(config)


def write_new_recipe(path: Path, config: dict) -> None:
    """Never replace an existing experiment's provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
