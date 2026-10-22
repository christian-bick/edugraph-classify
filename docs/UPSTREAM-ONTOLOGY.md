# Upstream ontology contract

This document explains the stable classifier-facing concepts of [`edugraph-ontology`](https://github.com/christian-bick/edugraph-ontology).

Current runtime (2026-10-03): **v0.30.0**, aligned with dataset v0.30.0-03 for the [completed Secure A40 baseline](RUNPOD-FULL-A40-20261002.md). The earlier Fireworks learning smoke remains pinned to dataset v0.30.0-02 and the same ontology version. The public eligibility/relations APIs remain compatible. Unlike the earlier v0.27 client-only migration, subsequent releases contain semantic changes; historical experiments require their pinned environment. The prior review below records the older migration context; the [refinement plan](REFINEMENT-PLAN.md) records new classifier-side evidence questions without redefining upstream semantics.

Reviewed against release **v0.27.0** on 2026-09-13. Its authored ontology is unchanged from v0.26.0. See [the v0.27.0 adoption notes](ONTOLOGY-UPGRADE-0.27.md), [the v0.26.0 semantic migration](ONTOLOGY-UPGRADE-0.26.md), and [the dataset upgrade](DATASET-UPGRADE-0.26.0-01.md).

## 1. Purpose and representation

EduGraph is an OWL/RDF ontology for describing educational content with small, reusable, observable descriptors and explicit relations. Its source is authored in Turtle and release artifacts include machine-readable ontology formats plus generated client libraries.

The classifier predicts core competency descriptors. Graph traversal, logical deduction, competency composition, and downstream recommendation remain deterministic ontology/application responsibilities where possible.

The ontology separates:

- **core descriptors**, which describe observable attributes of content;
- **competency descriptions**, which represent intersections of descriptors for recognizable applied competencies.

This separation avoids treating every useful Area-Scope-Ability combination as a new monolithic class label.

## 2. Core classes

All classifier labels belong to one of three subclasses of `CompetencyDescriptor`:

| Dimension | Stable meaning | Classifier interpretation |
|---|---|---|
| `Area` | A specific domain of knowledge and understanding within a field | What mathematical or subject-matter knowledge the task concerns |
| `Scope` | An observable context that affects abstraction, variation, generalization, complexity, or measurable difficulty | The setting, constraints, representation, range, or conditions under which the task operates |
| `Ability` | A general mental attribute that is trainable and applicable across fields | What cognitive action or capability the learner-facing task elicits |

Abilities are deliberately domain-general. An Area-Ability pairing common in one dataset is a statistical correlation, not an ontological rule that the Ability belongs only to that Area.

`CompetencyDescription` remains a `CompetencyEntity`, but since v0.24.0 it has no ontology-wide descriptor-presence or cardinality requirements. Applications define their own completeness policy. Several descriptors in one dimension apply conjunctively; no dimension or primary Ability is mandatory. The classifier keeps all three JSON fields for a stable output shape and permits empty arrays.

The released visual dataset exposes descriptor local names rather than `CompetencyDescription` individuals. Classifier code groups those descriptors by dimension using the pinned ontology package.

## 3. Identifier contract

Ontology individuals use globally unique IRIs under the EduGraph namespace. Generated TypeScript and Python libraries expose `Area`, `Scope`, and `Ability` enums whose values are full individual IRIs, definitions, and relation helpers.

The public dataset shortens an individual IRI such as:

```text
http://edugraph.io/edu/IntegerMultiplication
```

to its local name:

```text
IntegerMultiplication
```

Do not infer dimension from spelling, prefixes, or a manually maintained list. Resolve the local name through the exact pinned `Area`, `Scope`, and `Ability` catalogs. Treat enum-qualified forms such as `Area.IntegerMultiplication` as programming-language notation, not serialized dataset identifiers.

Names are identifiers, while `rdfs:isDefinedBy` and related annotations carry meaning for humans and models. Prompts or label embeddings that include definitions must use the definitions from the same ontology version as the identifiers.

## 4. Structural relations

| Relation | Inverse | Meaning and classifier use |
|---|---|---|
| `partOf` | `hasPart` | Constituent to containing field; navigation only, with no inheritance or label substitution |
| `specializes` | `specializedBy` | Narrower form of the same concept; a pure specialization chain supports the broader claim |
| `structures` | `structuredBy` | Shared structural parent of both relations; combined navigation does not establish inheritance |

For example, `Square specializes Rectangle`, whereas `HalfCircle partOf CircularShapes` does not imply `Circle`. Use the pinned client's `specializes_transitive` for broader concept claims and `structures_transitive` only for structural context. A path containing `partOf` must not be used to remove a supposedly redundant broader label.

Under upstream [ONT-E7](https://github.com/christian-bick/edugraph-ontology/blob/v0.27.0/docs/content-evidence.md#ont-e7--label-observable-descriptors-not-organizational-nodes), a descriptor with constituent children (`hasPart`) is organizational and cannot be a direct content label. Specialization children alone do not disqualify it: `Rectangle` and `ProofMethod` remain eligible. Inspect the complete pinned graph, including generated inverse relations, rather than a filtered tree. Eligibility does not establish image evidence.

The catalog retains all descriptors for lookup and derives `eligible_labels` through the released Python client's `is_label_eligible()` API. `labels()` rejects organizational labels instead of dropping them or replacing them with arbitrary children. Optional specialization-derived labels remain separate from explicit targets and metrics; the current converter performs no closure or ancestor pruning.

## 5. Logical constraint relations

`constrains` groups descriptor-level constraint relations. The two classifier-relevant specializations are:

| Relation | Directional meaning | Safe use |
|---|---|---|
| `A implies B` | Truth of A logically guarantees truth of B | Derived logical closure, constraint validation |
| `A contradicts B` | Requirements represented by A and B conflict wholly or partially | Recorded conflict diagnostics; hard rejection requires separately established instance-level incompatibility |

`impliedBy`, `contradictedBy`, and `constrainedBy` are inverse directions.

Logical implication is not the same as taxonomy. For example, one numerical bound can imply a looser bound without being a taxonomic child in the content-labeling sense.

Implication is directional: a looser predicted bound does not establish a tighter one. Choosing tighter explicit labels requires visible evidence and the versioned annotation convention. A relation lookup cannot recover information the prediction omitted, and a mathematically true broad bound may still disagree with the released explicit target.

Recorded compatibility checks should consider implication closure before contradiction: an implied descriptor may have a recorded conflict even when the original pair has no direct edge. The pinned client's helpers implement that graph policy. They do not compute exact mathematical intersections or prove that every flagged pair is impossible for a particular task.

**Correction verified against v0.30.0 on 2026-10-03:** [ONT-R2](https://github.com/christian-bick/edugraph-ontology/blob/db5d9541223880ba63e2eaa6f62d8db3186a9e2c/docs/relations.md#L58) explicitly permits partial conflict. `NumbersSmaller10` and `NumbersLarger10` both include magnitude 10 while recording a contradiction. A task whose relevant quantities all have magnitude 10 satisfies both definitions. The [upstream handoff](UPSTREAM-ANNOTATION-HANDOFF-20261003.md#42-recorded-contradictions-are-not-always-impossible-conjunctions) therefore requires an explicit instance-validity contract before using these relations for hard range correction. This corrects the earlier summary's strict-exclusion interpretation; no baseline output or metric has changed.

Hard constraints require explicit or validly inferred facts with the corresponding hard-exclusion semantics; a partial-conflict edge alone is insufficient. Absence of a relation does not prove incompatibility, prerequisite, or equivalence.

## 6. Progression and composition relations

Progression relations describe objective structure between competency spaces, but they are not label entailment:

| Relation | Stable meaning | Important non-meaning |
|---|---|---|
| `A expands B` | Understanding A is based on an understanding of B while extending the competency space | Does not prove that a task explicitly demonstrates B |
| `A inverts B` | A is an inverse concept of B; `inverts` is a specialization of `expands` | Does not make the two labels interchangeable |
| `A integrates B` | A is synthesized in parts using B | Does not require adding or removing B from every A-labeled task |
| `A translates B` | A represents B from another perspective, notation, or visualization; `translates` specializes `integrates` | Does not assert identical explicit evidence or exact label equivalence |

Their inverse directions are `expandedBy`, `invertedBy`, `integratedBy`, and `translatedBy`.

The ontology intentionally avoids a fuzzy universal `requires`/`hasPrerequisite` relation. Progression edges provide a deterministic structural skeleton that statistical systems may use as evidence for learning-path hypotheses, hard-negative construction, graph embeddings, or candidate neighborhoods.

For direct classification, these relations are priors and analysis features. They must not silently add labels to or remove labels from the model's explicit prediction.

## 7. The special case of `integrates`

`A integrates B` says that B participates as a component in synthesizing A. Whether B belongs in the explicit label set depends on the independent evidence in the classification unit:

- if B is merely instrumental machinery inside a task whose pedagogical target is A, the minimal explicit set normally omits B;
- if the isolated task independently asks the learner to demonstrate B, B can be explicitly valid as well;
- the relation alone is insufficient to decide between those cases.

This is why the atomic-input assumption is important. With one pedagogically coherent task per image, explicit evidence can be judged from the task rather than from unrelated exercises elsewhere on a worksheet.

## 8. Dimension independence and correlation

Area, Scope, and Ability are modeled as independent reusable dimensions so that descriptors can transfer across competencies. Real datasets will still show strong correlations among them.

Classifier systems must distinguish three sources of compatibility:

1. **Ontology facts:** class membership, taxonomy, implication, and contradiction.
2. **Implementation facts:** which combinations a particular generator/view system can realize.
3. **Empirical priors:** which combinations happen to co-occur in a dataset release.

Only the first category is inherently portable beyond the generating system. An unseen combination is not invalid merely because it has low or zero empirical frequency. A generative baseline may learn correlations jointly; a later graph-aware model should apply them as soft priors unless the ontology provides a true hard constraint.

## 9. Direct versus derived label views

Classifier storage and evaluation should preserve at least two views:

```text
explicit labels
  = labels independently supported by the artifact

derived labels
  = deterministic closure or graph facts computed from explicit labels
```

Examples of derived information include specialization ancestors and logically implied bounds. Structural context reached through `partOf` is navigation metadata. Keeping the views separate ensures that:

- exact-set metrics measure what the model was asked to predict;
- consumers can choose their preferred closure depth;
- ontology upgrades can recompute derivations without rewriting historical model output;
- graph post-processing gains are observable rather than attributed to the neural model.

Progression neighbors are neither explicit labels nor logically derived labels. Store them as related/candidate metadata if used.

## 10. Canonicalization boundary

The ontology supplies facts; `edugraph-classify` supplies the explicit-output policy. A deterministic canonicalizer may:

1. validate identifier membership in the pinned ontology;
2. resolve Area, Scope, and Ability dimensions;
3. remove exact duplicates and impose serialization order;
4. flag redundant specialization ancestors; remove them only under an explicitly validated, versioned output policy consistent with the relevant gold contract;
5. report implication-aware recorded conflicts; reject a set only under a separately validated instance-incompatibility rule, respecting partial conflicts and shared numeric endpoints;
6. compute optional derived closures without mixing them into explicit output.

It may not:

- map an unknown label to a similar-sounding known label;
- infer dimension from the identifier's English name;
- use `partOf` or combined `structures` paths as inheritance or an ancestor-removal rule;
- use `expands`, `integrates`, `inverts`, or `translates` as automatic add/drop rules;
- treat observed training co-occurrence as ontology truth;
- hide changes by discarding the raw prediction.

Every evaluation record should retain raw structured output, parsed explicit labels, canonical labels, validation findings, and any separately derived view.

The current baseline performs no ancestor pruning or logical graph repair. Its `invalid_output_rate` measures the implemented JSON/schema, membership, dimension and duplicate checks; it is not a comprehensive logical-consistency or visible-evidence metric. The [post-run graph audit](REFINEMENT-PLAN.md#graph-validation-and-explicit-serialization) found no recorded contradiction conflicts, while some exact predictions and their released gold both retain specialization ancestors. Such cases require upstream convention review before introducing cleanup rules. Eligibility, recorded descriptor compatibility, instance satisfiability and support in the image remain separate questions.

## 11. Versioning and client libraries

Ontology releases publish machine-readable ontology assets and generated language bindings. The TypeScript and Python clients expose:

- the `Area`, `Scope`, and `Ability` catalogs;
- definitions;
- direct relation lookups;
- transitive traversal helpers;
- logical compatibility/deduction helpers.

Classifier implementation should prefer a pinned generated client or a normalized snapshot derived from one immutable release. It should not scrape the ontology browser, use an unversioned checkout, or combine identifiers from one release with definitions/relations from another.

Every run manifest must record the exact ontology release or package artifact and a content hash of the normalized label/definition/relation snapshot used by conversion and evaluation.

The current `descriptor-semantics-v2` snapshot hashes sorted dimension/name/IRI records, definitions, generated direct relation views, and direct-label eligibility. Relation targets are deduplicated and sorted by full IRI. The manifest records `normalized_snapshot_format` alongside `normalized_snapshot_sha256`. Historical hashes without that format describe the old identifier-only snapshot and must not be compared as semantic fingerprints. The released wheel SHA-256 in `uv.lock` independently pins the entire package. Generated relation views include inverse and superproperty facts; this snapshot is not an authored-RDF or general OWL-reasoner snapshot.

Since v0.27.0, Python exposes typed immutable snapshot contexts, authored and entailed relation access, traversal, eligibility, compatibility, and the existing deduction helpers. The classifier uses the released eligibility helper now; the context API can support later canonicalization and derived-label work without local graph indexing. Optional RDF parsing is not installed because this repository consumes the bundled released ontology rather than authoring Turtle. Ontology authoring validation and assessment remain TypeScript-only.

## 12. Assumptions that are intentionally not stable

Do not hard-code:

- the number of labels in any dimension;
- individual label names or definitions outside versioned fixtures;
- taxonomy depths or branching factors;
- the presence of a particular relation between two current labels;
- current Area-Scope-Ability co-occurrence frequencies;
- a fixed set of candidate models or prompt definitions;
- the idea that every sample has exactly one descriptor per dimension.

Generate enums, schemas, dimension maps, graph closures, and candidate vocabularies from the pinned release.

## 13. Authoritative upstream references

- [`edugraph-ontology` README](https://github.com/christian-bick/edugraph-ontology/blob/main/README.md)
- [Core OWL schema](https://github.com/christian-bick/edugraph-ontology/blob/main/core-schema.ttl)
- [Ontology editing and semantic guidelines](https://github.com/christian-bick/edugraph-ontology/blob/main/DOCS_ONTOLOGY.md)
- [Ontology design rationale](https://github.com/christian-bick/edugraph-ontology/blob/main/DESIGN.md)
- [Generated TypeScript client usage](https://github.com/christian-bick/edugraph-ontology/blob/main/libraries/typescript/README.md)
- [Generated Python client usage](https://github.com/christian-bick/edugraph-ontology/blob/main/libraries/python/README.md)
