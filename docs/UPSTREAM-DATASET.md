# Upstream dataset contract

This document transports the stable parts of the [`edugraph-content`](https://github.com/christian-bick/edugraph-content) contract that classifier code needs. It intentionally omits sample counts, label frequencies, covered standards, current release numbers, and implementation-module inventories. Those are release-specific facts and must be profiled from the pinned artifact.

The upstream repository remains authoritative. If this summary and a pinned upstream release disagree, the pinned release and its documentation win.

## 1. Purpose and ownership

`edugraph-content` produces synthetic, visually rendered educational tasks with ontology labels known before generation. It owns:

- competency-target authoring from education standards;
- mathematical problem generation;
- learner-facing question and solution rendering;
- final label resolution;
- deterministic train/validation assignment and deduplication;
- artifact-level visual quality assurance;
- publication of the merged dataset.

`edugraph-classify` consumes released artifacts. It must not reproduce generator/view matching, reinterpret curriculum standards, or silently repair upstream gold labels. Suspected gold-label defects should be reported upstream and tracked separately from model errors.

## 2. Label-driven generation

The pipeline is label-driven rather than retrospectively labeled:

```text
competency label conjunction
          |
          v
compatible generator + view
          |
          v
canonical mathematical payload
          |
          v
learner-facing image
          |
          v
validated released sample
```

The target label set constrains the mathematics, relevant context, and requested learner action before pixels are rendered. A released sample is therefore intended as positive evidence for the complete conjunction in `tags`, not as an image that happened to receive plausible labels afterward.

All labels on one row mean **A AND B AND ...**. They are not alternatives, ranked suggestions, or a menu from which the classifier should select one.

## 3. One row is one classification unit

For classifier purposes, every released image is one standalone pedagogical task and one multilabel example. Upstream documents, worksheets, or videos that contain several independent tasks belong in an earlier segmentation stage; they must not be passed to this classifier as if they were atomic.

The upstream renderer has two modes:

- `solution: false` identifies a question/unsolved task rendering;
- `solution: true` identifies a solution/worked or filled rendering.

Question and solution rows are separate classification examples. Within generation they belong to the same structural exercise, share the competency target, and are kept or removed together during deduplication. They may nevertheless be independent deterministic problem draws; public consumers must not assume that a question row and a solution row depict the identical numeric instance or can be paired by filename.

The `solution` flag is a meaningful evaluation slice. Problem-only classification can require more mathematical inference than classifying a worked solution, so metrics must preserve this distinction.

## 4. Stable public artifact shape

The released union is organized into named splits with a `metadata.jsonl` file and referenced image files:

```text
dataset/
  train/
    metadata.jsonl
    <file_name>
  validation/
    metadata.jsonl
    <file_name>
```

Each nonblank JSONL line has the compact training-facing shape:

```json
{
  "file_name": "<path relative to the split directory>",
  "tags": ["<ontology-local-name>", "<ontology-local-name>"],
  "solution": false
}
```

Field semantics:

| Field | Stable meaning | Consumer rule |
|---|---|---|
| `file_name` | Path to the image relative to the containing split root | Resolve as a relative path; treat the name itself as opaque |
| `tags` | Complete explicit ontology-label set asserted for the image | Treat as a mathematical set, even if serialized in deterministic order |
| `solution` | Whether the image is the solution-mode rendering | Preserve for training manifests and sliced evaluation |

The public projection deliberately omits operational fields such as generator, view, target ID, sample key, random seed, fingerprints, curriculum-standard provenance, and deduplicated target associations. Do not infer these fields by parsing `file_name`; filenames are locators, not a supported semantic API.

The released `tags` values are shortened ontology individual identifiers such as `Addition`, not enum-qualified strings such as `Area.Addition` and not full IRIs. Resolve their dimensions through the exact pinned ontology package rather than naming conventions.

## 5. How final labels are owned

The generation architecture separates canonical mathematics from learner-facing task behavior:

- Generators own the abstract mathematical payload and the Area/Scope claims established by that mathematics.
- Views own the learner-facing projection and every Ability claim, because an Ability is only true when the final task actually asks the learner to exercise it.
- A view may also establish a presentational Area or Scope when the representation itself is educationally meaningful, but it must preserve the evidence supplied by the generator.

The final screenshot must make every supplied label reasonably identifiable and defendable through visible or necessary textual evidence. Label names or hidden payload fields are not substitutes for rendered evidence.

Upstream authoring selects the most specific ontology label that remains true and avoids adding ancestors merely to restate the taxonomy. Consequently, consumers should ingest `tags` as the explicit gold set and compute any ancestor or logical closure as a separate derived view. Derived labels must not be mixed into exact-set training targets or scored as additional explicit gold labels.

## 6. Target breadth versus rendered specificity

Education-standard targets may be broader than the concrete generator/view capability that satisfies them. Matching is one-directional: an equal or more-specific generated capability may satisfy a broader target through the ontology's `partOf` hierarchy.

The published row contains the labels resolved for the concrete artifact, including runtime choices, rather than merely copying an abstract standard statement. This distinction matters because classifier training is about what the image demonstrates, not every broader curriculum concept under which the task can be indexed.

## 7. Union, overlap, and deduplication

The released dataset is a derived union of non-isolated education-standard datasets. Standards overlap by design, so adding another standard does not imply that every generated task becomes another public row.

The merge follows declared standard precedence and removes duplicate configured tasks. Within the view scope used by the pipeline, it also prevents validation mathematical content from duplicating train content. Question and solution modes of one structural exercise are retained or dropped as a unit so deduplication cannot leave a partial exercise behind.

Important consequences for classifier work:

- repeated label sets are normal and do not imply duplicate images;
- two images with the same `tags` may present different valid tasks or visual forms;
- one retained physical sample can operationally evidence several overlapping standard targets even though public metadata contains only one row;
- curriculum provenance and task identity cannot be reconstructed from the compact public projection;
- official splits should be preserved unless a new split can be built from richer upstream identity/fingerprint metadata.

## 8. Split semantics

Train is the primary generated artifact. Validation is generated with knowledge of train so that mathematical content already present in train is excluded from validation within the pipeline's view-scoped fingerprint comparison. Split-integrity tooling also rejects configured-task redundancy within the corresponding view scope.

Split assignment and sample randomness are identity-derived, not dependent on filesystem enumeration order. Upstream release checks audit leakage and redundancy, but classifier ingestion should still fail closed on missing files, duplicate public paths, malformed rows, or accidental cross-split byte duplication.

Because the public schema omits structural identities and fingerprints, arbitrary row-level resplitting can separate related renderings or reintroduce semantic leakage. Prefer the official train/validation split. If a test set is introduced, derive it upstream or obtain the operational grouping fields needed to split complete task groups.

## 9. Determinism and visual validation

Canonical generation pins the rendering environment and derives all sample entropy from structural identity. The output is intended to be reproducible for the same complete source and renderer identity.

Before release, canonical artifacts pass static integrity checks and artifact-level VQA. VQA evaluates the final image against the complete claimed label conjunction and the view's observable contract. This is a gold-data quality gate, not a guarantee that every downstream classifier will find the image equally easy.

Do not replace upstream artifact validation with classifier confidence. A confident model can be wrong, and a low-confidence model does not establish a gold-label defect.

## 10. Dataset and ontology version alignment

Every dataset release is tied to the ontology version used to generate and validate it. Labels, definitions, dimensions, and relations can change across ontology releases, so a dataset tag without its ontology provenance is not a complete experiment identity.

Every classifier run must record at least:

- immutable dataset repository and release/revision;
- dataset split and source-content hashes;
- ontology package/version used by that dataset;
- conversion code commit and output hashes.

Never validate an older dataset's labels against an unpinned latest ontology package. A label missing from the latest ontology may still be valid for the dataset release that contains it, while a reused local name may have changed meaning.

## 11. Required ingestion behavior

A correct dataset adapter should:

1. resolve an immutable dataset release and its ontology provenance;
2. parse every nonblank JSONL row strictly;
3. reject absolute or split-escaping `file_name` paths;
4. verify every referenced image exists and record its content hash;
5. treat `tags` as an unordered set and reject non-string values;
6. validate each tag's existence and dimension against the pinned ontology;
7. preserve the upstream split and `solution` flag;
8. serialize one canonical classifier target for each set;
9. profile release-specific counts, frequencies, cardinalities, and label-set coverage rather than hard-coding them;
10. retain source-row identity in the generated run manifest without committing provider-ready images or JSONL to Git.

Canonicalization may detect duplicate tags or redundant ancestors, but it must report any change to gold labels. Silent gold mutation would make model metrics incomparable to the released dataset.

## 12. Assumptions that are intentionally not stable

Do not encode any of the following in long-lived classifier logic:

- number of samples, labels, or exact label sets;
- label frequency or class balance;
- which standards, grades, generators, or views are covered;
- image dimensions or visual style;
- label cardinality per row;
- presence of every ontology label in every split;
- a pairing convention derived from filenames;
- release-specific validation thresholds or reports.

Those belong in generated data profiles and run manifests.

## 13. Authoritative upstream references

- [`edugraph-content` README](https://github.com/christian-bick/edugraph-content/blob/main/README.md)
- [`edugraph-content` technical documentation](https://github.com/christian-bick/edugraph-content/blob/main/DOCS.md)
- [Target-spec evidence and label rules](https://github.com/christian-bick/edugraph-content/blob/main/docs/target-spec.md)
- [Generator/view specification rules](https://github.com/christian-bick/edugraph-content/blob/main/docs/spec-general.md)
- [Published metadata projection](https://github.com/christian-bick/edugraph-content/blob/main/src/lib/dataset-merge.ts)
