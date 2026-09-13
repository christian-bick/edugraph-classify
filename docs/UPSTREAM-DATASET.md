# Upstream dataset contract

This document transports the stable consumer-facing contract of [`edugraph-dataset`](https://github.com/christian-bick/edugraph-dataset). It intentionally excludes release counts, label frequencies, current coverage, and internal implementation architecture.

The pinned upstream release remains authoritative. If this summary and a released artifact disagree, the released artifact and its documentation win.

**Release compatibility, checked 2026-09-13:** the current recipe pins v0.26.0-01 at `cee47a3b49503e2637759a8a0e59e071c271b415` together with ontology v0.26.0. Its actual metadata uses `labels`, although the pinned dataset README still says `tags`. The importer selects the field explicitly through `dataset.label_field`; old configurations without it retain the `tags` contract. It rejects missing, unexpected, or simultaneous fields without guessing or merging. See [the release audit](DATASET-UPGRADE-0.26.0-01.md) for measured compatibility and data-quality findings.

## 1. Purpose and ownership

`edugraph-dataset` publishes synthetic images of educational tasks with EduGraph ontology labels. It owns:

- released images and metadata;
- official dataset splits;
- the meaning of gold label sets;
- dataset and ontology version alignment;
- upstream integrity and evidence validation.

`edugraph-classify` consumes those released artifacts. It must not reinterpret curriculum sources or silently repair upstream gold labels. Suspected label defects should be recorded separately from model errors and reported upstream.

## 2. One row is one classification unit

Every released image is one standalone pedagogical task and one multilabel example.

For real-world material, this establishes the preprocessing boundary:

> One independent pedagogical task equals one classification unit.

Worksheets, documents, or videos containing several independent tasks must be segmented before classification. The direct-labeling model is not responsible for discovering task boundaries.

## 3. Stable public artifact shape

The released dataset provides named splits with a `metadata.jsonl` file and referenced image files:

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
  "labels": ["<ontology-local-name>", "<ontology-local-name>"],
  "solution": false
}
```

| Field | Stable meaning | Consumer rule |
|---|---|---|
| `file_name` | Image path relative to the containing split root | Resolve it as a relative path and treat the name itself as opaque |
| `labels` (`tags` in historical releases) | Complete explicit ontology-label set asserted for the image | Pin the field name in the experiment; treat its value as a mathematical set, regardless of serialized order |
| `solution` | Whether the image presents a solution rather than an unsolved task | Preserve it in manifests and evaluation slices |

The public contract does not expose curriculum provenance or an identity that links related rows. Do not extract semantics from filename components. Filenames are locators, not a supported metadata API.

Released labels are shortened ontology individual identifiers such as `Addition`, not programming-language expressions such as `Area.Addition` and not full IRIs. Resolve each label's dimension through the exact pinned ontology package rather than its spelling.

## 4. Label-set meaning

The dataset is label-driven: the labels are known constraints of the task, not retrospective guesses based on a finished image.

All labels on one row form a conjunction:

```text
A AND B AND C
```

They are not alternatives, ranked suggestions, or a list from which the classifier should select one.

Every asserted label is intended to be reasonably identifiable and defendable from visible or necessary textual evidence in the image. A label name appearing as decorative text is not, by itself, evidence for the corresponding competency.

The gold set describes the concrete artifact being classified. It should not be broadened merely because the task can also be indexed under a curriculum category or a taxonomic ancestor.

## 5. Explicit and derived labels

Upstream labels follow a minimal-explicit-label convention: use the most specific descriptor that remains true without repeating broader facts already carried by the ontology.

Classifier ingestion must preserve the released label array exactly as the explicit gold set. Organizational descriptors with `hasPart` children are ineligible under ontology v0.26.0; reject and report them rather than dropping them or substituting descendants. Specialization children alone do not disqualify a broader descriptor. Information computed mechanically from the ontology belongs in a separate derived representation:

```text
explicit gold labels
        |
        v
optional ontology closure
        |
        v
derived ancestors / implications
```

Derived labels must not be mixed into direct SFT targets or counted as additional explicit-label successes. Keeping the two representations separate also permits ontology closures to be recomputed after a version change without rewriting historical predictions.

## 6. Question and solution samples

The `solution` field defines two important cohorts:

- `false`: an unsolved question or task;
- `true`: a worked, filled, or otherwise solution-presenting task.

Each row stands on its own as a classification example. The public metadata does not provide a supported pairing identifier, so consumers must not assume that question and solution rows can be paired or that similarly named files contain the same mathematical instance.

Preserve the flag during training and report performance separately for both cohorts. Unsolved tasks can require more inference to identify the intended operation or procedure, while a presented solution can expose additional evidence.

## 7. Released union and duplicates

The public release combines content aligned to supported education standards. Overlap between standards is normal, and the released artifact removes duplicate content according to its publication policy.

Important consumer consequences:

- repeated exact label sets are expected and do not imply duplicate images;
- two images with identical labels may depict different valid tasks or representations;
- one image may be relevant to several curriculum statements even though the public row is singular;
- curriculum provenance cannot be reconstructed from the compact metadata;
- label-set equality is not image or task identity.

Classifier tooling should check for duplicate paths and bytes as an ingestion safeguard, but it should not discard examples merely because their label sets match.

## 8. Split semantics

`train` and `validation` are the official upstream splits. They are created and checked as part of dataset publication, including checks intended to prevent leakage and redundant task content.

Preserve these splits for the baseline. Arbitrary row-level resplitting can introduce semantic leakage or separate related material because the public schema intentionally omits richer grouping identity.

If a new held-out test set is needed, it should be published upstream or constructed from additional stable grouping metadata rather than inferred from filenames.

Classifier ingestion should still fail closed on:

- a path appearing more than once in a split;
- a missing referenced image;
- identical image bytes appearing across splits;
- malformed metadata;
- an unknown ontology identifier;
- a label whose dimension cannot be resolved or which is structurally ineligible as a direct label.

These checks detect packaging or ingestion mistakes; they do not redefine the upstream split policy.

## 9. Reproducibility and evidence validation

Released samples are produced reproducibly from a versioned source and undergo upstream validation before publication. Validation checks that each final image is a coherent standalone task and supports its complete asserted label conjunction.

This is a gold-data quality contract, not a guarantee that every downstream classifier will find every image equally easy. Model confidence must not be used as a replacement for upstream validation: a confident model can be wrong, and low confidence does not establish a gold-label defect.

Classifier code should hash every consumed image and generated metadata artifact so that experiments can be reproduced even if mutable local paths later point elsewhere.

## 10. Dataset and ontology version alignment

Every dataset release is tied to the ontology version used to label and validate it. A dataset revision without its ontology provenance is not a complete experiment identity.

Every classifier run must record at least:

- immutable dataset repository and release/revision, with its metadata label field;
- split names and source-content hashes;
- ontology release/package associated with the dataset;
- conversion code commit and generated-artifact hashes.

Never validate an older dataset against an unpinned latest ontology package. A label absent from the latest ontology can still be valid for the historical dataset release that contains it, while a retained local name may have changed meaning.

## 11. Required ingestion behavior

A correct dataset adapter should:

1. resolve an immutable dataset release and its ontology provenance;
2. parse every nonblank JSONL row strictly;
3. reject absolute or split-escaping `file_name` paths;
4. verify every referenced image exists and record its content hash;
5. read the configured `labels` or historical `tags` field as an unordered set and reject non-string or repeated values;
6. validate every label's existence, dimension, and direct-label eligibility against the pinned ontology;
7. preserve the official split and `solution` flag;
8. serialize exactly one canonical classifier target for each explicit set;
9. profile counts, frequencies, cardinalities, and coverage from the release rather than hard-coding them;
10. retain source-row identity in the run manifest without committing provider-ready images or JSONL to Git.

Canonicalization may detect duplicate tags or redundant ancestors, but it must report any change to gold labels. Silent mutation would make model metrics incomparable to the released dataset.

## 12. Assumptions that are intentionally not stable

Do not encode any of the following in long-lived classifier logic:

- number of samples, labels, or exact label sets;
- label frequency or class balance;
- standards or grades currently covered;
- image dimensions, format, or visual style;
- label cardinality per row;
- presence of every ontology label in every split;
- pairing conventions inferred from filenames;
- release-specific validation statistics.

Those belong in generated data profiles and run manifests.

## 13. Authoritative upstream references

- [`edugraph-dataset` repository](https://github.com/christian-bick/edugraph-dataset)
- [`edugraph-dataset` README](https://github.com/christian-bick/edugraph-dataset/blob/main/README.md)
- [Released dataset on Hugging Face](https://huggingface.co/datasets/christian-bick/edugraph-exercises)
