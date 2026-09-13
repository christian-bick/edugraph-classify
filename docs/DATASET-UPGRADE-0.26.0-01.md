# Dataset v0.26.0-01 upgrade

Verified 2026-09-13 against the public release. Dataset v0.26.0-01 resolves the earlier ontology-eligibility mismatch and cross-split byte-leakage blocker. Every released row converts under ontology v0.26.0 without modifying gold labels or official splits. One within-training duplicate with differing gold sets remains recorded below.

## Pinned identities and importer adaptation

- Dataset: [`christian-bick/edugraph-exercises`, v0.26.0-01](https://huggingface.co/datasets/christian-bick/edugraph-exercises/tree/v0.26.0-01), resolved through the [Hub refs API](https://huggingface.co/api/datasets/christian-bick/edugraph-exercises/refs) to `cee47a3b49503e2637759a8a0e59e071c271b415`.
- Ontology: `edugraph-py` v0.26.0, official wheel SHA-256 `596f7845de647ec23f8724b6c4aa99085846043cc5b9fd6350a455bd1c663b71`.
- Ontology semantic snapshot: `descriptor-semantics-v2`, SHA-256 `0e3b1e4c3d71c85891e19d45cd7ed6085ccd32829138ec753bf3d37baacb134a`.
- Recipe: [`edugraph-20260913-qwen35-9b-local-ddp-v1.json`](../experiments/edugraph-20260913-qwen35-9b-local-ddp-v1.json), using the full official splits, `direct-label` prompt v1, and `dimension-labels` schema v1.

The release metadata contains exactly `file_name`, `labels`, and `solution`; Hugging Face exposes the image reference as `image`. The pinned dataset card still documents `tags`, so the actual released metadata determines the adapter shape. `dataset.label_field: "labels"` explicitly selects the current contract. Historical configurations without that setting retain `tags`. Conversion rejects ambiguous fields, missing fields, malformed arrays, repeated labels, unknown identifiers, and organizational descriptors. Prepared manifests record the resolved field name. No heuristic rename, ancestor pruning, child substitution, or entailment is applied.

Historical experiment configurations and launch records retain their original identities. Their v0.21.0 ontology pins still fail the current version check before downloading data; reproduction requires their original code and lockfile.

## Full release audit

The production streaming loader and converter read every released image at the immutable revision. Validation included exact metadata shape, pinned paths, image decoding and hashing, label existence/dimensions/eligibility, duplicate paths, duplicate bytes, and deterministic dimension-aware targets.

| Split | Rows | Questions | Solutions | Distinct labels | Label assignments | Rows failing ontology validation |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,630 | 815 | 815 | 297 | 13,084 | 0 |
| Validation | 314 | 157 | 157 | 179 | 2,770 | 0 |

All 1,944 image references decoded successfully. There were no duplicate source paths and no identical image bytes across splits. Public metadata lacks a supported task-group identifier, so byte checks do not prove the absence of all semantic overlap. Structural eligibility also does not independently revalidate the visual evidence for every gold label; that remains upstream's responsibility.

| Artifact | SHA-256 |
|---|---|
| Released train metadata | `7c6f826188c414aed5c0951a8aac75ea8f55da7172d3429f196cd4024ee7df58` |
| Released validation metadata | `16eaa75c2dfb791a28ed3cabb0038bb17f290d482cf48068680a19f8b9940a64` |
| Converted train JSONL | `205babda9af83f1adc0ddde16cdc8ce2968960d6975236d0cdc2c7b1a56c6627` |
| Converted validation JSONL | `08e24f8b8cbf5cdb29bf4922d24092686af028180e236c4c9405634b75dd6438` |
| Source manifest | `39b052c227d67a36408e789c50f551d4f5fbcbde7266f6ab40f7ed332d71378d` |
| Profile | `888b26d7ea75eaf41cd19d1f814512f804f0b0476742952dbedbf715749182b6` |

Generated audit outputs are local, gitignored artifacts under `temp/dataset-v0.26.0-01/`. This was a conversion audit against the working implementation, not a prepared training run claiming a committed code identity. After committing the changes, `uv run edugraph-classify run prepare` regenerates the artifacts and records the clean code commit in a run manifest.

## Remaining gold-data observation

Two question images in the training split have identical bytes, SHA-256 `64bd883983634e299c7b04ae1bafc98dbb39c1730afbbdffb2185349afe68cff`, but differ in one explicit Area label:

| Path relative to `train/angle-arithmetic/` | Differing Area |
|---|---|
| `4-MD-C-7-additive-angle-measure-569e9916_angle-arithmetic_geometry-angle-arithmetic-inversion_inst-0_mode-Q.png` | `Addition` |
| `4-MD-C-7-unknown-angles-daa9d2b3_angle-arithmetic_geometry-angle-arithmetic-inversion_inst-0_mode-Q.png` | `Subtraction` |

Both gold sets also contain `AdjacentAngles`, `AngleCalculation`, `DegreeScale`, and `ProcedureInversion`. All these labels are ontology-eligible. This is a separate gold-set consistency observation for upstream review. The converter preserves both rows and reports one `conflicting_gold_label_groups` entry in the profile, following the existing within-split duplicate policy.

## Training boundary

The new default recipe pins `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, two RTX 3090s, two-process DDP QLoRA, microbatch 1 per GPU, and accumulation 2 (global batch 4). It records `local_docker` / `self_hosted` separately from the GCP storage and image-registry settings. GCP remains the provider for services surrounding training and receives completed model artifacts.

These settings can be prepared and recorded today. The strict local job schema, Docker lifecycle, shared GCS staging, distributed trainer changes, host preflight, and coordinated recovery/publication in the [local training plan](LOCAL-TRAINING.md) still need implementation. The existing `run launch` command rejects this local recipe because it launches only Fireworks jobs. No dataset upload, container publication, model upload, or training was performed for this upgrade.

## Repository verification

`uv run --locked --group dev pytest --cov=edugraph_classify --cov-report=term-missing` passed all 146 tests with 95.41% coverage. Tests exercise both metadata field contracts, malformed and ambiguous rows, eligibility rejection, unchanged cross-split byte protection, deterministic output across worker counts, the current recipe's manifest identities, and rejection of local manifests by Fireworks launch. `uv lock --check --offline`, `uv build`, and `git diff --check` also passed. No dependency changes were required.
