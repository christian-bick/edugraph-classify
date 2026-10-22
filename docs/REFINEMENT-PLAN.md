# Classifier refinement after the first full baseline

Updated 2026-10-03. This is the current research direction after the completed [Secure A40 baseline](RUNPOD-FULL-A40-20261002.md) and its subsequent error analysis. The [kickoff](PROJECT-KICKOFF.md) retains the architectural principles and earlier decisions. The studies below are proposed experiments, not demonstrated improvements or authorization to launch paid runs.

Retain direct Qwen3.8-27B classification with the compact shared system prompt, constrained label generation, language-only QLoRA, frozen vision/bridge, and generated evaluation as the reference. Prioritize observable label conventions and specific semantic distinctions. Dataset changes belong upstream in `edugraph-dataset`; definitions and relations belong in `edugraph-ontology`. Preserve the released gold and historical scores while investigating ambiguities.

**Accepted next priorities, 2026-10-03:** consistent annotations for realized-task numeric ranges and complete coverage of all legitimate, observable labels, without selecting facets merely for distinguishability. The [upstream annotation handoff](UPSTREAM-ANNOTATION-HANDOFF-20261003.md) supplies the two implementation analyses, source evidence and release acceptance gates. Dataset correction precedes another model or inference-cleanup experiment. The completeness goal is settled; precise quantity-domain rules, annotation interfaces and specific ambiguous cases still need upstream implementation decisions.

The [label-convention audit](LABEL-CONVENTION-AUDIT-20261003.md) inspected 91 distinct images, including 53 of the 69 final errors. Numeric ranges and abilities account for 79/156 label events, but combine supported model errors with annotation-policy issues. Historical observations below remain unchanged. This plan does not authorize gold mutation or another paid run.

## 1. Evidence and its limits

The reference is `edugraph-20261002-qwen38-27b-runpod-full-a40-v1`: dataset v0.30.0-03, ontology 0.30.0, two epochs, 1,654 training images and 828 updates. Its selected epoch-2 adapter scored 181/250 exact matches (72.4%), 96.44% label micro-F1 and zero invalid outputs on the final cohort. The run record contains immutable source, model, prompt, code and artifact identities.

The following additional findings come from offline analysis of those same saved predictions, not another model run:

| Observation | Implication for research |
|---|---|
| Of 69 nonexact images, 54 fail in only one dimension; 35 fail only scopes. Forty-nine need at most two label additions/removals. | Prioritize specific distinctions and omissions; many failures retain the main task labels. |
| Numeric-range labels contribute 45 of 156 false-positive/false-negative label events, on 27 images; 14 images have only range errors. | Numerical specificity is the largest concrete audit target. One substitution counts as two label events. |
| All 250 final gold label sets already occur in training. Training covers 300/680 eligible labels; final gold covers 176/680. | The baseline tests familiar combinations on held-out images. It does not establish unseen-label or novel-combination generalization. Equal label sets do not prove duplicate tasks or leakage. |
| Epoch 2 corrects 15 selection images but regresses five; 28 remain exact and 16 remain wrong. | Additional training has tradeoffs; assess gains and regressions together. The 16-image training diagnostic cannot establish global overfitting. |
| The two epochs agree on 28/64 selection label sets, all correct; their 36 disagreements contain all 21 epoch-2 errors and 15 correct predictions. | Checkpoint disagreement is a candidate review signal, not calibrated confidence. It has not been validated on an independent cohort. |
| Recorded implication-aware contradiction checks find no conflicts in the final gold or predictions. | Recorded graph conflicts are not an observed leading source of recoverable errors. The ontology permits partial descriptor conflicts, so this diagnostic is not a general test of instance satisfiability or mathematical correctness. |

Exact-set match also reflects label cardinality: images with at most eight gold labels score 97/117 exact (82.9%), versus 84/133 (63.2%) with nine or more. Their label F1 remains similar, approximately 96.7% and 96.3%. Report cardinality slices alongside overall exact match rather than interpreting every failed set as wholesale task misunderstanding.

There is no demonstrated general question-versus-solution difficulty ordering: selection favors solutions, while the final cohort favors questions. These are different, unpaired image cohorts. Image-token slices likewise do not show a monotonic resolution-related failure pattern. Neither result establishes a need to increase image resolution or train vision layers.

All cluster comparisons are exploratory, may overlap and often have small denominators. Do not treat images as statistically independent task groups without supported grouping metadata, infer groups from filenames, or infer causes from correlations alone.

## 2. Label clusters and the evidence audit

| Cluster | Measured evidence | Question to resolve |
|---|---|---|
| Numerical specificity | All 12 missed `NumbersSmaller10` labels (16 gold occurrences) and four missed `NumbersSmaller5` labels (seven occurrences) coincide with extra `NumbersSmaller20`. `NumbersSmaller10` has 106 training occurrences. | Define the realized-task quantity domain and strongest supported bounds consistently; preserve broader generation constraints separately. |
| Required cognitive action | `ProcedureInversion` is missed in 5/13 occurrences; `ProcedureExecution` has seven extra predictions and no misses across 91 gold occurrences. | Does the visible task ask for execution, explanation, interpretation or working backward? |
| Operation versus strategy | All three missed `Subtraction` labels coincide with extra `SubtractionPlaceValuePartitioning`. | Is that particular strategy independently visible, or is only the operation supported? Their `expands` relation does not authorize substitution. |
| Fraction-related concepts | The ontology-defined fraction cohort has 14/24 complete exact matches, with mixed subtype, reference-frame and ability errors. | Which distinctions require better evidence, definitions or coverage? A single range correction cannot explain this entire cluster. |
| Operand detail | `SingleDigitSmallestOperand` is missed in all three final occurrences; `TwoDigitLargestOperand` in 2/7. | Correct independently verified annotation omissions across generators, then evaluate model omissions against complete targets; these small baseline supports alone do not measure completeness. |

Ontology-defined cohorts select images by gold membership in the pinned node or its `structures` descendants, for navigation only. They do not add closure to gold labels. Positive controls include metric-weight cohorts at 25/25 complete exact matches and metric-volume cohorts at 10/10. Individual labels with no false positives or omissions include `DecimalNumbers` (21 gold occurrences), `Division` (17) and `AreaCalculation` (15). Preserve these controls when testing targeted changes; these counts do not imply universal accuracy.

### Numerical-bound observability

The pinned definitions of `NumbersSmaller5`, `NumbersSmaller10` and `NumbersSmaller20` use inclusive absolute bounds on task inputs/results. The tighter bounds `imply` looser ones; they are not a `specializes` chain. A broader predicted bound cannot be refined by reversing implication without additional image evidence.

Targeted visual inspection found a training image showing `8 - 7 = 1` with gold `NumbersSmaller20`, and another showing `5 - [blank] = 2` with the same bound. A final image showing `6 + 2 = 8` instead has gold `NumbersSmaller10`, which the model replaces with `NumbersSmaller20`. The completed audit found that inspected release-source code assembles configured labels before sampling values, while its visual QA checks support without requiring a unique label set. This establishes a plausible implementation mechanism for broad-but-true labels; exact source-to-artifact build provenance remains unverified. It is not a proven dataset-wide defect or a license to relabel.

For reproducibility, the first training locator is `train/arithmetic-ops-pairs/1-OA-C-6-fluency-f50a9669_arithmetic-ops-pairs_operations-boxes-inversion_inst-0_mode-S.png`; the final locator is `validation/arithmetic-ops-pairs/1-OA-C-6-fluency-7ca8ac54_arithmetic-ops-pairs_operations-boxes_inst-0_mode-S.png`, both at the pinned dataset revision in the run record. These strings identify files; they are not supported grouping metadata.

Reviewers should first inspect the image without the model answer, then compare its defensible labels with the pinned definitions and released gold. Record clear model errors, ambiguous evidence, serialization-policy disagreements and suspected upstream defects separately. Resolve the intended numerical convention upstream before reweighting these examples or generating more of them. Do not silently relabel historical data or weaken its primary metric.

### Graph validation and explicit serialization

Eight final gold sets contain both `LiquidVolumes` and its specialization ancestor `VolumeMeasurement`; all eight predictions match their gold exactly. A ninth prediction has a `CommonDenominator`/`FractionNumbers` ancestor pair on an already incorrect image. Blanket ancestor pruning would break eight exact answers. Flag such cases for upstream policy review and retain raw predictions; do not assume a generic graph cleanup improves the released task.

Eligibility, schema validity, logical consistency and visible evidence are different checks. The current zero invalid-output count measures the implemented structural/vocabulary checks. The separate graph audit uses recorded implication/contradiction facts and pure specialization paths, not a general mathematical solver. The pinned ontology's `contradicts` can express partial conflict: inclusive upper/lower bounds share endpoints even when the client reports an exclusion. The [handoff](UPSTREAM-ANNOTATION-HANDOFF-20261003.md#42-recorded-contradictions-are-not-always-impossible-conjunctions) explains the required distinction before hard correction. Progression relations, including `integrates`, remain contextual evidence and never automatic add/drop rules.

## 3. Ordered research plan

| Priority | Study | Evidence required before adopting a change |
|---|---|---|
| 1 | Implement the [two upstream annotation priorities](UPSTREAM-ANNOTATION-HANDOFF-20261003.md): consistent realized-task ranges and complete observable label coverage. Track remaining semantic cases separately. | Shared quantity/applicability rules; independent fact coverage; separate generation claims and realized annotations; boundary, replay, cache and release checks; preserve the original evaluation alongside any corrected-release result. |
| 2 | Create controlled contrast examples after resolving the relevant convention. Hold layout and unrelated labels constant while crossing a numeric boundary, moving the unknown operand or changing the requested cognitive action. | Better performance on independent contrasts and the broader development cohort, with reported regressions on positive controls. Balance cluster coverage; do not blindly oversample uncertain or rare examples. |
| 3 | Establish fresh development/challenge data and a separate final confirmation set upstream, with supported task/group identities. Include new combinations, broader label coverage and unfamiliar visual styles. | Separate results for known/new combinations, labels and styles; related renderings stay together. Preserve official released splits and do not reconstruct group identity from filenames. |
| 4 | Compare direct prediction with targeted pinned definitions or a short verifiable evidence step for confused concepts. | Exact-set gains at an acceptable latency/cost, with final labels scored separately. Record prompt/schema changes; preserve candidate recall and an escape from incomplete candidate lists. |
| 5 | Compare an additional epoch, learning-rate changes and targeted sampling one factor at a time on development data. | Generated predictions showing corrected and introduced errors, cluster/support counts, positive controls and uncertainty across repeat runs where needed. Keep vision and bridge frozen. |

Fresh development data can be prepared alongside the upstream audit; it must be available before using the later experiments to select a winner. A changed learning rate or prompt is a new experiment, not a compatible continuation under the current resume contract. Follow the [Runpod recovery rules](RUNPOD-TRAINING.md#recovery-and-one-epoch-continuation) and retain complete provenance.

As a supporting study, validate checkpoint-disagreement review on independent data. Measure the fraction sent for review and the errors captured at that review budget, including errors on which models agree. Combine disagreement with coverage/diversity so one cluster does not consume the annotation budget. Existing raw outputs do not supply calibrated per-label probabilities; agreement or sequence likelihood must not be presented as such.

## 4. Evaluation rules for follow-up work

The 250-image cohort was excluded from the original checkpoint selection, so the historical baseline result stands. Its errors have now informed research choices. Treat subsequent reuse as exploratory or regression testing; it is no longer a blind final assessment for changes selected from these findings. Confirm the selected follow-up on a fresh assessment set kept separate from annotation, prompt and hyperparameter decisions.

Keep exact-set match primary, with order-independent labels and raw versus canonical output retained. Report precision/recall/F1, label macro-F1, per-label support, per-dimension/view/frequency/cardinality slices, invalid output, latency and cost. Separate broad-but-compatible predictions from mathematically unsupported predictions diagnostically without silently awarding partial credit in the primary metric. Do not restrict predictions to combinations observed in training.

Define the hypothesis, development cohorts, expected regression checks and comparison procedure before launching a study. Each new recipe pins dataset/ontology, prompt/schema, code, model, training settings and evaluation. The plan does not grant automatic model promotion, deployment, dataset upload or paid compute authorization.

## 5. Methods deferred by the current evidence

| Method | Current decision and condition to revisit |
|---|---|
| Additional Markdown/format repair | Retain constrained decoding and validation; zero final structural errors make further repair a low priority. |
| Hard graph correction or automatic ancestor pruning | Audit conventions first. No recorded final contradictions were found, and pruning conflicts with some released gold. Soft ontology neighborhoods remain a candidate for retrieval or reranking. |
| Full ontology vocabulary in every prompt or blanket reasoning | Keep the compact baseline; test only relevant definitions or verifiable evidence on a specific confusion. Extra reasoning may also add unsupported labels. |
| Larger models, blanket resolution increases or vision unfreezing | No established causal evidence for these changes. Frozen vision/bridge is the current training requirement; keep it throughout this plan. |
| RL, preference learning or set-reward optimization | Defer until preferences and evidence conventions are reliable and simpler controlled studies identify a remaining objective mismatch. |
| Discriminative multilabel heads or graph-aware decoders | Revisit for serving latency, calibrated per-label scores or a demonstrated generative-model limitation. Include calibration, imbalance and ontology-migration costs; these are not yet required for accuracy. |

## 6. Evidence locations

The [full-run record](RUNPOD-FULL-A40-20261002.md) identifies the checksummed GCS model bundle containing the baseline, epoch and final predictions. Prepared examples retain original image locators and hashes. The derived local analysis is under ignored `reports/runpod-full-a40-20261002/edugraph-20261002-qwen38-27b-runpod-full-a40-v1/`:

- `analysis-clusters/summary.json`, `errors.json`, `ontology-cohorts.json` and `graph-audit.json`;
- `analysis-dynamics/results.json`;
- `analysis-manual-spotcheck.json` with inspected image identities and observations.

These local analysis files are not checked into Git or newly published GCS artifacts. The numerical findings above are the versioned design summary; the underlying baseline predictions remain in the immutable model bundle. No new training, label mutation or inference deployment was performed for this analysis.
