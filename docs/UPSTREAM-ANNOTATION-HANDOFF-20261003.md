# Upstream handoff: consistent ranges and complete observable annotations

Prepared 2026-10-03 for `edugraph-dataset` and `edugraph-ontology`. This is an implementation-ready analysis, not an implemented dataset migration. It records the user's accepted direction and proposes the concrete contracts, work sequence and acceptance evidence needed upstream.

## 1. Accepted outcome and ownership

The next dataset work has two priorities:

1. **Consistent range annotations:** equivalent observable numerical tasks receive equivalent bounds, using one documented rule for which quantities count and how the bounds are selected.
2. **Complete observable annotations:** every legitimate, observable descriptor is represented, directly or by a documented sound derivation. Annotation must not omit an independent fact because it is elementary, does not distinguish a curriculum target, or was never added to a generator's capability schema.

Completeness applies across Areas, Scopes and Abilities. It does not mean adding every visually suggestive label: the pinned definition and the task's actual evidence must support it. Nor does it mean adding organizational nodes or interpreting `integrates`, `involves`, `expands` or `partOf` as logical entailment. Independent facts cannot be omitted under the name of minimal serialization. Only facts recoverable through approved logical/specialization rules can be represented by a compact basis.

The two detailed work packages are:

- [Range annotation analysis](UPSTREAM-RANGE-ANNOTATIONS-20261003.md): quantity domain, boundary cases, generator/view mechanisms, canonical bound selection and regression cases.
- [Label completeness analysis](UPSTREAM-LABEL-COMPLETENESS-20261003.md): concrete omitted facets, capability asymmetry, observation contracts, missing-label validation and rollout checks.

`edugraph-dataset` owns annotation generation, render evidence, validation, provenance, splits and release. `edugraph-ontology` owns definitions, eligibility, relations and client semantics. `edugraph-classify` preserves released gold and historical metrics, then consumes a corrected version. Other labeling issues remain a separate case-by-case queue; they do not need to block implementation of resolved systematic rules.

## 2. Evidence and source identities

| Evidence | Immutable identity | Interpretation |
|---|---|---|
| Dataset actually used in training | Hugging Face `christian-bick/edugraph-exercises`, v0.30.0-03, `9b509867e898490615be3f59bc2f31fac429389c` | Source of the 1,968 prepared images and their original gold |
| Matching dataset GitHub release source | [`d1920909e34bfe368cc22a4aea6e591690db796d`](https://github.com/christian-bick/edugraph-dataset/commit/d1920909e34bfe368cc22a4aea6e591690db796d) | Source mechanism inspected for the historical release; not a verified source-to-HF build receipt |
| Ontology used in the baseline | v0.30.0, [`db5d9541223880ba63e2eaa6f62d8db3186a9e2c`](https://github.com/christian-bick/edugraph-ontology/commit/db5d9541223880ba63e2eaa6f62d8db3186a9e2c) | Meaning of the historical labels |
| Dataset implementation inspected for the handoff | [`374055902a5105d39408ba43e72c3dbea56ce7e5`](https://github.com/christian-bick/edugraph-dataset/commit/374055902a5105d39408ba43e72c3dbea56ce7e5) | Clean local upstream checkout at inspection; proposed implementation anchors use this snapshot |
| Ontology used by that implementation | [`275616e7115c61c4d2080baf0e16ddf71a961163`](https://github.com/christian-bick/edugraph-ontology/commit/275616e7115c61c4d2080baf0e16ddf71a961163), preview `0.30.0-pre.8.275616e7115c` | Dataset [package pin](https://github.com/christian-bick/edugraph-dataset/blob/374055902a5105d39408ba43e72c3dbea56ce7e5/package.json#L71); not the classifier's baseline ontology |

The local checkout named `edugraph-content` is the `edugraph-dataset` repository. No current generated preview artifacts were used to assign source provenance or grades to historical images. Image locators are opaque identifiers, not grouping or curriculum metadata. Existing visual judgments are targeted assistant review, not an independently adjudicated population defect estimate.

The [first audit](LABEL-CONVENTION-AUDIT-20261003.md) examined 91 distinct images, including 53 of 69 final-assessment errors. Numeric-range disagreement accounts for 45 of 156 label error events across 27 images: 18 bound substitutions, seven omitted gold bounds and two additional bounds absent from gold. Both selected bounds were defensible under the recorded quantity interpretation in the 18 substitutions. These counts establish priority, not the attainable accuracy gain from a dataset fix.

The subsequent completeness investigation found, for example, that `6 + 2 = 8` omits both applicable single-digit operand descriptors while `6 × 1` and `8 × 3 = 24` area models include both. `443 + 353 = 796` omits the corresponding three-digit descriptors. Their exact locators, image digests and original annotations are recorded in the completeness package. Digit facets occur on 64 of 1,968 released rows; this is an emission frequency, **not** a completeness percentage because not every row has relevant operands.

History supports incremental feature adoption and uneven generator coverage, rather than a universal failure to backfill lower grades: the operand-digit family was requested during Grade 4 work, added with simple and advanced variants together, and explicitly adopted in Grade 1 place-value work the next day. The remaining omissions should be audited by applicable facts and task/view contracts, not by grade heuristics.

## 3. Shared architectural finding

The current pipeline performs these distinct operations as one annotation path:

```text
requested target -> matched capabilities -> selected configuration labels
                                                       |
                                                       v
                                       labels already fixed before draw
                                                       |
                                                       v
                                      mathematical payload -> view -> image
```

In [`resolvePlannedConfigurations`](https://github.com/christian-bick/edugraph-dataset/blob/374055902a5105d39408ba43e72c3dbea56ce7e5/src/lib/planned-generation.ts#L62), the label union is assembled at lines 74–75; `generatePlannedDraw` generates the mathematical instance at line 91. That is sufficient to preserve requested guarantees. It cannot, by itself, annotate every additional fact that depends on the realized instance or final projection.

The current [IMPL-G3 rule](https://github.com/christian-bick/edugraph-dataset/blob/374055902a5105d39408ba43e72c3dbea56ce7e5/docs/implementation-generator.md#L40) deliberately keeps labels outside the generator and treats draws such as `23 + 18` versus `31 + 7` as ordinary instance variation. Those draws can have different operand-digit facts under the accepted completeness goal. This is an intentional contract extension to design, not evidence that all existing modules violated their original rules.

**Recommended architecture:** retain label-driven generation, then add a separate deterministic annotation stage using structured mathematical evidence and the actual prepared task projection. Preserve three objects:

| Object | Purpose | Must not be conflated with |
|---|---|---|
| Requested and selected generation claims | Matching, sampling constraints, replay and curriculum-target provenance | Complete annotation of the final image |
| Realized observable facts and their evidence | Complete instance/view-specific description | Every hidden value in a generator payload |
| Exported explicit labels plus separately reproducible derivations | Public gold under a versioned serialization policy | A union of all labels mentioned by targets, generators or views |

The annotation stage belongs in shared orchestration or typed annotation adapters. Keep generator mathematics free of label emission and keep renderers free of ontology-label-dependent repair. If a view makes a seeded choice that changes the observable task, the annotation stage must consume the same resolved projection or an explicit evidence receipt; it must not resample that choice independently. Question and solution artifacts need their own evidence decisions. They are not necessarily paired draws in the existing pipeline.

Adding every facet to every generation schema is an alternative, but creates combinatorial configuration growth and couples sample diversity to annotation completeness. Rejection-sampling into exact range bins can also change the content distribution. Prefer deriving facts from the realized task, while retaining generation constraints for curriculum coverage and deliberate sampling.

## 4. Integration decisions that must be explicit

### 4.1 Tighter ranges do not satisfy today's generic matcher automatically

Dataset [SPEC-1](https://github.com/christian-bick/edugraph-dataset/blob/374055902a5105d39408ba43e72c3dbea56ce7e5/docs/spec-general.md#L13) uses equality or pure `specializes` ancestry for capability matching. Numeric bounds are related through `implies`. [`assertResolvedTargetCoverage`](https://github.com/christian-bick/edugraph-dataset/blob/374055902a5105d39408ba43e72c3dbea56ce7e5/src/lib/label-contracts.ts#L72) uses that existing capability ancestry.

Simply replacing selected `NumbersSmaller100` with realized `NumbersSmaller20` before the existing assertion can therefore fail coverage, although the tighter predicate proves the broader one. Keep the planned claim set and its existing validation separate. Specify an additional evidence-to-target entailment proof for realized annotations, including the exact approved relation paths. Audit deduplicated target associations and the coverage explorer against that distinction; do not silently broaden generic matching or rewrite selection receipts.

### 4.2 Recorded contradictions are not always impossible conjunctions

Both inspected ontology versions define numerical bounds inclusively. When every relevant task quantity has absolute value 10, `NumbersSmaller10` and `NumbersLarger10` are both true, yet the graph records a contradiction. This is intentional descriptor-level partial conflict in [ONT-R2](https://github.com/christian-bick/edugraph-ontology/blob/db5d9541223880ba63e2eaa6f62d8db3186a9e2c/docs/relations.md#L58), not a new preview-only change or proof that one label is false.

The client [`incompatible`](https://github.com/christian-bick/edugraph-ontology/blob/db5d9541223880ba63e2eaa6f62d8db3186a9e2c/libraries/typescript/OntologyContext.ts#L197) follows recorded implication/contradiction exclusions rather than computing interval intersections. Before enforcing strongest lower/upper annotations, upstream must distinguish descriptor conflict diagnostics from hard instance invalidity. A documented interval-aware decision or a distinct hard-incompatibility contract is needed. Do not change inclusive definitions, suppress a supported bound, or treat every `incompatible` result as mathematical impossibility just to pass the existing helper.

This also corrects an earlier classifier-documentation assumption that all `contradicts` edges were strict logical exclusion. Historical graph-audit counts remain measurements of recorded conflicts; the original model scores are unchanged.

### 4.3 Completeness needs its own validation identity

[`validateLabelChecks`](https://github.com/christian-bick/edugraph-dataset/blob/374055902a5105d39408ba43e72c3dbea56ce7e5/src/lib/vqa-policy.ts#L88) accepts exactly the supplied label list. It cannot report an additional omitted label as a structured label check. Preserve support validation, and add a separate completeness result with reviewed candidate/facet coverage and evidence. Missing capability declarations or unimplemented evaluators must appear as uncovered work, not automatically as “not applicable.”

The existing [VQA key](https://github.com/christian-bick/edugraph-dataset/blob/374055902a5105d39408ba43e72c3dbea56ce7e5/src/lib/vqa-cache.ts#L145) incorporates image bytes, label/checklist context and validation policy. Extend dependency and cache identity to include the annotation policy, annotation implementation, reviewed candidate inventory and relevant ontology records, including descriptors absent from the old gold. An old passing support check is not evidence of completeness. Pixel reuse may be valid for annotation-only changes, but annotations, audit results, validation records and exports must invalidate correctly.

### 4.4 Keep the public interface and historical artifacts deliberate

The current [public exporter](https://github.com/christian-bick/edugraph-dataset/blob/374055902a5105d39408ba43e72c3dbea56ce7e5/src/lib/dataset-merge.ts#L53) projects `file_name`, `labels` and `solution`. Keep rich evidence/provenance in versioned operational metadata or an explicitly published sidecar unless upstream intentionally versions the public interface. Record the annotation policy and source-to-release build provenance in the release manifest. Recheck content/task deduplication, identical-image label conflicts and train/validation leakage after migration.

## 5. Suggested upstream work sequence

| Work item | Owner and deliverable | Acceptance gate |
|---|---|---|
| A. Annotation contract | Dataset: adopt realized-task ranges and complete observable facts; separate these from target-selection policy | Worked positive, negative and boundary examples; no grade/generator-dependent omission rule |
| B. Range semantics/API reconciliation | Ontology + dataset: fix the interpretation boundary between inclusive truth and descriptor-level conflict; specify quantity roles | Equal endpoints and overlapping domains handled explicitly without suppressing true facts |
| C. Shared annotation/evidence stage | Dataset: pure typed resolvers, view projection evidence and provenance alongside existing generation receipts | Replay determinism; no extra random draw; existing target claims still proved; no annotation from hidden nonnecessary payload data |
| D. Coverage rollout | Dataset: inventory all eligible descriptor families and actual producer/view combinations; implement deterministic numeric facts and reviewed view facts first | Each relevant family has evidence-backed output or an explicit unresolved status; no “implemented schema = entire candidate universe” shortcut |
| E. Independent completeness QA | Dataset: support checks plus missing-label checks; blindness to selected gold where practical; cache/dependency updates | Seeded omissions are caught, correct additions accepted, unsupported additions rejected, unresolved critical cases block a complete-release claim |
| F. Versioned release | Dataset: migration report, manifest, regression corpus and new gold | Stable image identities where retained, complete label-diff report, integrity/split audits, exact source/ontology/policy provenance |
| G. Classifier comparison | Classifier: consume the corrected release and later rerun the fixed baseline recipe | Original score retained; fresh confirmation data kept outside audit/tuning; any paid run separately authorized |

For C–E, first compute proposed annotations in a **local shadow audit** of preserved instances/projections: report additions, removals, ambiguous cases and evidence without overwriting gold. This is a way to review the dataset change, not a model inference-cleanup experiment. Use old instances when complete replay provenance exists; otherwise regenerate under a new version and report content changes separately from annotation changes. Do not infer missing replay or grouping information from filenames.

## 6. Release acceptance checklist

- One annotation policy gives the same independent facts for equivalent evidence regardless of curriculum target, grade or generating module. Differences caused by actually different learner actions or visible evidence remain legitimate.
- Upper and lower range fixtures cover endpoints, hidden necessary answers, zeros, negatives, fractions, measurement units, distractors, intermediate quantities and representation capacity; unresolved domains are explicit.
- Removing a known supported digit/range/view label makes a completeness test fail. Injecting an unsupported label makes a support test fail. Existing exact predictions are included as controls because incomplete gold can still be predicted exactly.
- Every eligible descriptor family has a reviewed applicability route or a documented applicability exclusion. Unknown/unimplemented is not a negative label or an exemption from the full-coverage goal.
- The canonical explicit set and its approved closure preserve all accepted facts. Unknown labels, organizational nodes, proofs with no supported starting fact and unsupported progression-based additions fail checks.
- Requested target claims, selected generation bindings and realized annotations remain individually inspectable. A stronger realized bound can prove a requested broad bound without falsifying the original selection receipt.
- Annotation and ontology changes invalidate all dependent annotation/completeness artifacts; unrelated pixels are not needlessly regenerated. An old support-only VQA cache cannot pass the new completeness gate.
- Export, deduplication, target associations and official splits are validated together. Same-image differing gold is diagnosed rather than silently unioned or arbitrarily selected.
- The migration report separates changes to pixels, task grouping, definitions, explicit labels and derivations. It reports resolved and unresolved cases per facet/view with denominators, not an unqualified “complete” claim from aggregate frequency.
- The old v0.30.0-03 release and 72.4% exact-set baseline remain intact. Fresh independent assessment data are prepared before using the next comparison for model selection.

No upstream edits, gold mutations, paid validation, training, uploads or inference deployment were performed to prepare this handoff. Specific strategy, fraction-evidence and solution-observability cases remain linked in the detailed analyses and the original audit rather than being silently settled by these systemic rules.

## 7. Handoff verification

The three proposals received an independent source/policy review. Offline checks verified the linked immutable Git blobs and line bounds, all six published example locators/image hashes/original-label excerpts, and the 23-file release/current comparison (21 identical files). This verifies reproducible references, not independent human adjudication of image semantics or remote URL availability.

Local reproduction uses `uv run --no-sync python temp/upstream-annotation-handoff/verify.py` with the existing baseline artifacts and sibling Git repositories. Its checksummed document/reference report is under ignored `reports/upstream-annotation-handoff-20261003/verification.json`. The proposals themselves contain the source pins, image links, measured facts and acceptance specifications needed for an upstream reader without those local scratch files. `git diff --check` also passed; no production code was changed or production test suite run for this documentation task.
