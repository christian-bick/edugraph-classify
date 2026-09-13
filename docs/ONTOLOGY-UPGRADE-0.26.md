# Ontology v0.26.0 migration assessment

Assessed 2026-09-13. The classifier previously installed v0.21.0. GitHub's latest published release is [v0.26.0](https://github.com/christian-bick/edugraph-ontology/releases/tag/v0.26.0), published at 11:33:38 UTC from commit `b9199b3928a10892ee04c8c2a1d7008c9356259f`.

The official Python wheel is pinned in `pyproject.toml` and `uv.lock`, with SHA-256 `596f7845de647ec23f8724b6c4aa99085846043cc5b9fd6350a455bd1c663b71`. No unrelated dependency version changed. Historical experiment configurations, launch records, dataset identities, and prompt identities retain their original values.

## Upstream changes acknowledged

Comparison of the published v0.21.0 and v0.26.0 Python clients found:

| Dimension | v0.21.0 descriptors | v0.26.0 descriptors |
|---|---:|---:|
| Area | 278 | 283 |
| Scope | 326 | 352 |
| Ability | 125 | 129 |
| Total | 729 | 764 |

These are release audit observations, not constants used by classifier logic. There are 37 added identifiers and two removed identifiers, with no dimension moves among retained names. Removed identifiers are `AbsoluteValue` and `AxiomDefinition`; new concepts include `AbsoluteNumberMagnitude`, `AxiomIdentification`, and `AxiomFormalization`. They must not be treated as automatic aliases. Four retained names have changed definition text: `Circle`, `DecimalDivisorShift`, `ErrorCorrection`, and `Visualization`. Generated relation records differ for 720 retained descriptors, including new structural superproperty views; that count does not mean 720 independently authored semantic changes.

The main changes across the intervening releases are:

- **Structural semantics:** `partOf` now describes constituent membership without inheritance. `specializes` carries broader-concept inheritance; `structures` combines both for navigation. The inverse helpers distinguish `has_part`, `specialized_by`, and `structured_by`. Only a pure specialization path supports a broader claim. [Structural contract](https://github.com/christian-bick/edugraph-ontology/blob/v0.26.0/docs/structure.md).
- **Direct-label eligibility:** descriptors with constituent children are organizational. A descriptor with specialization children can still be directly observable. This is the same rule for all three dimensions and must use the complete graph. [ONT-E7](https://github.com/christian-bick/edugraph-ontology/blob/v0.26.0/docs/content-evidence.md#ont-e7--label-observable-descriptors-not-organizational-nodes).
- **v0.24.0:** adds `SixthFractions`; removes the three descriptor-presence equivalences from `CompetencyDescription`. There is no ontology-wide minimum, maximum, or required dimension. Completeness is an application policy. [Release](https://github.com/christian-bick/edugraph-ontology/releases/tag/v0.24.0).
- **v0.24.1–v0.24.2:** adds `CircularShapes` as the organizational parent of `Circle`, `HalfCircle`, and `QuarterCircle`; clarifies `Circle` and moves three fraction strategies under `FractionStrategies`. Structural navigation must be regenerated. Half/quarter circles do not inherit `Circle`. [v0.24.1](https://github.com/christian-bick/edugraph-ontology/releases/tag/v0.24.1), [v0.24.2](https://github.com/christian-bick/edugraph-ontology/releases/tag/v0.24.2).
- **v0.25.0:** adds 20 justification Scopes covering proof methods, evidence basis, coverage, and error control. The root is organizational; families such as `ProofMethod` and evidence-supported specializations can be explicit labels. These additions do not provide training examples by themselves. [Release](https://github.com/christian-bick/edugraph-ontology/releases/tag/v0.25.0).
- **v0.25.1:** corrects four shape-to-angle progression links to `integrates` and removes a redundant nanometer assertion already derived from `translates`. Refresh progression-dependent results; do not turn those edges into label entailment. [Release](https://github.com/christian-bick/edugraph-ontology/releases/tag/v0.25.1).
- **v0.26.0:** adds shared TypeScript snapshot, RDF, and assessment APIs, with parser-independent core entry points and RDF term-aware diagnostics. Its four authored Turtle files are unchanged from v0.25.1. Python remains a descriptor/relation client, so this repository has no TypeScript import migration or reasoner dependency to adopt. [Migration guide](https://github.com/christian-bick/edugraph-ontology/blob/v0.26.0/libraries/typescript/README.md#6-supplied-snapshots-and-portable-imports).

## Implemented classifier adaptations

`OntologyCatalog` still resolves dimensions from upstream enums and rejects unknown and duplicate identifiers. It now computes direct-label eligibility from the generated `hasPart` view and rejects organizational labels with an explicit diagnostic. It retains the full catalog for lookup; it never changes the gold label set to make it pass. There are 86 organizational descriptors in this release, leaving 678 structurally eligible candidates, before evidence validation.

The new `descriptor-semantics-v2` fingerprint includes definitions, sorted relation targets, and eligibility alongside identifier/dimension/IRI records. Prepared manifests record its format. This fixes the earlier identifier-only hash, which could miss a relation-only or definition-only change. Historical hashes remain historical; regenerate snapshots and caches for new runs rather than rewriting old manifests.

No output-schema cardinality change was necessary: `LabelSet` and `dimension-labels-v1.json` already permit empty arrays. Existing explicit-label conversion performs no ancestor pruning or entailment, so there was no `partOf` traversal algorithm to replace. Tests cover organizational rejection across dimensions, valid broader specialization labels, new scopes, empty dimensions, inheritance boundaries, semantic hashing, and rejection of a historical version under the current package.

The current prompt does not embed ontology definitions or a candidate vocabulary. Its bytes and version remain unchanged. A future graph-aware prompt, constrained vocabulary, logical validator, or canonicalizer must use eligible candidates, versioned definitions, specialization-only redundancy rules, and separate explicit/derived outputs. The Python catalog is not an ontology authoring validator or a complete logical prediction validator.

## Dataset compatibility

The [public dataset refs](https://huggingface.co/api/datasets/christian-bick/edugraph-exercises/refs) showed no v0.26.0 release on the assessment date. The most recent tag is `v0.22.2-01`, at immutable commit `5e81395636f1f83c522b04866e7e8aa341d37521`. Its [pinned card](https://huggingface.co/datasets/christian-bick/edugraph-exercises/blob/5e81395636f1f83c522b04866e7e8aa341d37521/README.md) ties dataset versions to the ontology used for generation.

An audit downloaded only the four pinned metadata JSONL files, without loading images or evaluating their evidence:

| Dataset release / split | Metadata label field | Rows | Rows containing a v0.26.0 organizational label |
|---|---|---:|---:|
| v0.21.0-01 / train | `tags` | 1,602 | 362 |
| v0.21.0-01 / validation | `tags` | 356 | 108 |
| v0.22.2-01 / train | `labels` | 1,560 | 326 |
| v0.22.2-01 / validation | `labels` | 318 | 84 |

All local names in these files still resolve in v0.26.0, but existence alone is insufficient. Organizational labels include `AngleMeasurement`, `LengthMeasurement`, `NumericComparison`, and `ShapeIdentity`. `Circle`, whose definition changed since v0.21.0, is also present. These observations assess compatibility with the new ontology, not correctness under each dataset's original ontology.

The newest dataset's actual metadata uses `labels`, despite its README still documenting `tags`. The current importer deliberately implements the old `tags` contract and will reject the new shape. The metadata files are reproducibly identified by these SHA-256 values:

| Release / split | SHA-256 |
|---|---|
| v0.21.0-01 / train | `6c8124cd4b253a820ba6151f9e929507bdb50bae52da58c3985e1dc00740ed21` |
| v0.21.0-01 / validation | `9272c8d218289c3d2d1a1004cc7a52f0ee292fac43a21e951beaad5f046d56bd` |
| v0.22.2-01 / train | `f07d91c4beeedfd7fe3c80259993f62909adfc1581acb85fe68dd53768e9ee46` |
| v0.22.2-01 / validation | `a467a9910967d64a4356650d477acde5db668bb6c6f07f228c0a283e38cba903` |

Before a new v0.26.0 training experiment:

1. Have the upstream dataset owner review affected labels against each image and publish an aligned release with official splits. Do not drop organizational labels or substitute their children automatically.
2. Pin the new release commit and ontology wheel hash together under a new experiment/run ID. Old experiments remain reproducible with their original code commit and `uv.lock`; the current version guard rejects them before downloading data.
3. Implement and test the new release's explicit metadata-field contract, including rejection of ambiguous/conflicting fields. Recheck dataset provenance rather than changing only `ontology.version` in an old recipe.
4. Regenerate conversion artifacts, semantic fingerprints, and profiles; rerun image integrity and cross-split leakage checks. This metadata audit does not establish whether the earlier byte-leakage problem is fixed in the newest dataset.
5. Reevaluate exact-set, dimension, frequency, invalid-output, and view metrics under the new label policy. Keep historical raw predictions and evaluation settings intact; measure any later post-processing changes separately.

There is intentionally no new runnable v0.26.0 experiment pointing at either older dataset. Dependency support is updated; the aligned training-data release remains an upstream prerequisite.

## Verification

The official wheel version and lockfile hash were checked, both published Python releases were compared locally, and all four metadata files were audited at immutable revisions. The complete local command `uv run --locked --group dev pytest --cov=edugraph_classify --cov-report=term-missing` passed all 127 tests with 95.38% coverage; `uv lock --check --offline` and `uv build` also passed. The CLI reports historical ontology mismatches with configuration-error exit code 2 before dataset access. No provider operation, image upload, or training run was launched.
