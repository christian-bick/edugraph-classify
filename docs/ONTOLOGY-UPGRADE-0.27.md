# Ontology client v0.27.0 adoption

Adopted 2026-09-13 from the official [`v0.27.0`](https://github.com/christian-bick/edugraph-ontology/releases/tag/v0.27.0) release at commit `bd68e4ecd4f36688dbaf122838315ab72bfa0ae8`. The pinned Python wheel SHA-256 is `5c519aad8cc5370a70ad58697c99c90ab7e3f77d155a2a5bb403a996f14ef160`.

## Compatibility decision

The four authored Turtle sources are byte-identical to v0.26.0. The classifier still resolves 764 descriptors, identifies 678 direct-label candidates, and produces the same `descriptor-semantics-v2` SHA-256, `0e3b1e4c3d71c85891e19d45cd7ed6085ccd32829138ec753bf3d37baacb134a`. There are no identifier, dimension, definition, relation, or eligibility changes and therefore no label migration.

The current dataset remains release v0.26.0-01 at its immutable Hugging Face commit. Its recipe now records `source_ontology_version: 0.26.0` while `ontology.version` identifies the v0.27.0 package used for conversion. This preserves the dataset's generation provenance and records the actual runtime authority. Historical recipes retain their original ontology versions and require their matching code and lockfiles.

## Adopted API

`OntologyCatalog` now delegates ONT-E7 structural eligibility to the released `is_label_eligible()` helper instead of interpreting the generated `hasPart` relation locally. Dimension grouping remains based on the released `Area`, `Scope`, and `Ability` enums, and the classifier-owned normalized fingerprint continues to cover definitions, deterministic relation views, and eligibility.

The release also provides typed immutable ontology contexts, authored versus entailed relation access, traversal, compatibility, deductions, supplied-snapshot decoding, and optional RDF parsing. These APIs remove the need for future classifier code to build graph indexes or reimplement implication-aware compatibility. They do not change the current explicit-label conversion policy. The optional RDF extra is not installed because classifier runs consume the bundled released snapshot; ontology authoring validation and assessment remain upstream TypeScript responsibilities.

## Verification

The dependency was updated with `uv`, leaving the package and lockfile synchronized. `uv run --locked --group dev pytest --cov=edugraph_classify --cov-report=term-missing` passed all 151 tests with 95.11% coverage. `uv lock --check --offline` and `uv build` also passed. A direct runtime check confirmed package version 0.27.0, 764 descriptors, 678 eligible labels, and the unchanged normalized snapshot hash. No dataset was uploaded and no provider or training operation was launched.
