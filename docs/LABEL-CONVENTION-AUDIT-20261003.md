# Label-convention audit, 2026-10-03

**Subsequent decision:** the user confirmed complete coverage of legitimate, observable labels and one consistent realized-task range rule as the next dataset priorities. The [upstream handoff](UPSTREAM-ANNOTATION-HANDOFF-20261003.md) supersedes the open policy recommendations below where that goal resolves them. This document retains the original audit observations and metrics; neither the released gold nor the baseline score has been changed.

Completed the first audit in the [refinement plan](REFINEMENT-PLAN.md#3-ordered-research-plan). The largest issue is the distinction between a **defensible label** and a **uniquely specified explicit label set**. Numeric ranges and cognitive-action labels account for 79 of the baseline's 156 missing/extra label events. Some are clear model errors; others require a more precise annotation convention before targeted retraining can be interpreted.

This is a classifier-side evidence audit and a proposed upstream decision package, not upstream approval or a corrected dataset. Released gold, predictions and the historical 181/250 exact score remain unchanged. No model calls, training, paid VQA, dataset uploads or upstream edits were performed.

## Audit protocol and coverage

1. Reproduce all 250 final predictions against the original prepared examples: 2,115 true-positive, 73 extra and 83 missed labels; 69 nonexact images. Labels are sets, independent of order.
2. Rank conventions by affected label events and images. Select all numeric-range error images; all ability errors; all Subtraction errors; and all errors involving fraction subtype, reference frame or the two observed operand-digit labels. Selection uses actual labels and pinned ontology relations, never filename-derived task families.
3. View each selected original image and record its task, visible quantities, requested action and solution evidence. Reviewers recorded image observations before revealing this pass's individual prediction differences where practical. Prior baseline exposure and label-based selection mean this was **not blinded independent adjudication**. Training controls were also label-stratified or purposively selected.
4. Compare evidence with ontology definitions, the released gold and the prediction. Keep supported model misses, evidence ambiguity, explicit-serialization disagreements and any suspected upstream defects distinct. Mixed cases retain their individual uncertainties. A label can be true but still be an exact-target error.
5. Compare training and exact-prediction controls; inspect immutable upstream policy/source snapshots. Cross-check consequential cases in a second assistant review. One initially suspected missing-explanation defect was withdrawn after full-image reinspection revealed its explanatory footer; it is a model omission. The ledgers retain this correction.
6. Verify reviewed image hashes against both the prepared examples and run manifest; save per-image observations, source snapshots, impact counts and a review gallery.

**91 distinct images were inspected:** 53 of the 69 final error images, 14 exact final controls and 24 training controls. Numeric controls select eight `NumbersSmaller20`, four `NumbersSmaller10` and four `NumbersSmaller5` training rows by ascending SHA-256 of `convention-audit-v1:` plus the opaque example ID. Other controls include all six training examples carrying `SubtractionPlaceValuePartitioning`. This is targeted coverage, not a random prevalence estimate or a complete release audit. The remaining 16 final error images have quantitative diagnostics but no new visual adjudication here.

All image judgments are provisional assistant assessments. No inter-rater reliability or population defect rate is claimed. The reused final cohort remains exploratory for follow-up choices.

## Measured impact

One substitution contributes one extra and one missed label. “Only errors” counts complete predictions whose disagreements are confined to the named labels; it is an oracle diagnostic, not an expected gain from changing a convention. Image counts overlap and cannot be added.

| Convention / diagnostic | Error images | Label events | Images with only these errors |
|---|---:|---:|---:|
| Numeric-range selection and inclusion | 27 | 45 | 14 |
| Cognitive-action / Ability selection | 20 | 34 | 10 |
| Subtraction replaced by place-value partitioning | 3 | 6 | 2 |
| Proper versus general fraction label | 3 | 5 | 0 |
| `SingleFrameOfReference` inclusion | 5 | 5 | 2 |
| Operand-digit detail | 4 | 5 | 2 |

The first two categories account for **50.6% of all label events**. This does not mean half the errors are annotation defects. For example, five supported operand-detail omissions have straightforward image evidence and are good model-improvement targets.

## 1. Numeric bounds: highest-priority convention decision

The 27 images separate into three distinct patterns:

| Pattern | Images | Events | Finding |
|---|---:|---:|---|
| Different selected bounds | 18 | 36 | Observed/necessary task quantities fit both gold and predicted bounds under the recorded quantity interpretation. |
| Omitted gold bounds | 7 | 7 | Five lower-bound and two upper-bound omissions; the gold predicates are supported. |
| Extra numeric bounds outside gold | 2 | 2 | Numeric bounds fit a fraction task and a length-difference task but are not selected in gold. |

Sixteen substitutions replace `NumbersSmaller5` or `NumbersSmaller10` with `NumbersSmaller20`. Two run in the other direction: `2 × 6 = 12` has gold `NumbersSmaller100` and predicted `NumbersSmaller20`; seven tens producing 70 has gold `NumbersSmaller120` and predicted `NumbersSmaller100`. Thus neither “always tighten the prediction” nor “always preserve the broader bound” reproduces released targets.

Examples in the ledger:

- **N05:** `6 + 2 = 8`, gold at most 10, prediction at most 20. Both predicates fit, but the prediction loses specificity relative to the target.
- **N03:** the explicitly requested known-fact/commutative relationship uses 2, 6 and 12; gold at most 100, prediction at most 20. Here the prediction is tighter.
- **N25:** seven tens and result 70; gold at most 120, prediction at most 100. Both fit even if the coefficient 7 is included.
- **N26/N27:** two ten-frames have capacity 20 but contain one/seven dots; gold at most 10, prediction at most 20. The actual count and representation capacity must not be conflated without a stated rule.
- **N31, training control:** comparison of a cylinder's two flat faces with a sphere's zero flat faces retains gold at most 20. The control demonstrates broad-but-true labels in training independently of the model's errors.

The quantity rule itself needs precision: actual inputs and necessary results versus frame capacity, distractor-pool size, place-value coefficients, and fraction numerators/denominators. No disputed numeric predicate was shown false under the documented observations; this is not a proof about every possible interpretation or the rest of each prediction.

### Why the current checks can accept these targets

The pinned dataset [visual checklist](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/src/visuals/views/checklist.md#L15-L20) tests the supplied labels for support, permits nonunique labels and allows an uncertain judgment to pass. It does not establish completeness or a unique explicit answer set.

The inspected [generation code](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/src/lib/planned-generation.ts#L62-L93) assembles labels from configured generator/view capabilities before sampling the instance. A generated value can satisfy a tighter bound while retaining the selected broader bound. The [numeric-range decision](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/docs/plan/numeric-range-bounds.md#L12-L38) governs truth across inputs and results; it does not specify a canonical tightest-bound output for every realized image.

**Recommended upstream decision:** define an image-observable applicability and specificity rule for numeric facets. If the task is image-only classification, a canonical choice among true bounds is preferable to recovering an unshown generation setting. Specify when the facet applies, which values enter it, and how the explicit upper/lower labels are selected. If intended instructional range is instead the target, it needs distinct semantics and sufficient visible evidence. Resolve this before oversampling range errors or changing gold. This audit does not enact either policy.

## 2. Abilities: main action, instrumental actions and solution evidence

This is the second-largest category, with 34 events across 20 images. It combines actual model misses with conventions that are not exclusively determined by literal label definitions.

**Primary versus instrumental action:** one task asks to complete sums and explain why they work, but gold keeps only `ProcedureUnderstanding`; predictions add execution and reading. An associativity explanation explicitly requests and displays written prose, but an extra `TextualArticulation` is penalized, whereas another explanation task includes that label. These cases require a consistent rule for which performed actions are the annotation target, not a rule that reading text automatically adds `TextualReception`.

**Inversion in solutions:** four of five missed `ProcedureInversion` labels are completed equations whose formerly unknown operand is green. No blank or inverse working remains. This is an intentional visual convention: training contains the same pattern, and the inspected [renderer](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/src/visuals/views/operations/arithmetic-boxes-view.tsx#L35-L58) distinguishes answered inputs from forward results by position and styling. These are weak-evidence/convention-learning cases, not demonstrated gold defects. The fifth miss, **A06 `4 − [blank] = 1`**, directly exposes the unknown input and is a clear model error. Exact controls show the model can recognize other explicit unknown-input questions and solved stories that retain their question.

**Action hidden by a completed artifact:** A24 shows an analog clock and the green answer `10:15 p.m.`. It does not clearly retain whether the learner was asked to construct a clock or read one. The source locator is not admissible image evidence for selecting visual articulation over interpretation/reception.

**Clear misses remain:** the counting solution in A11 explicitly explains one-to-one counting and why the last number is the total; missing `ProcedureUnderstanding` is an error. Other cases visibly request concept classification, deriving a unit relationship or expressing a representation as equations. Do not explain these away as convention issues.

**Recommended upstream decision:** label the elicited cognitive action and retain multiple abilities when independently required by the task. Publish examples distinguishing central assessed actions from incidental channels. For solution images, preserve the original unknown/request or show enough working to support the claimed action. [Ontology evidence rules](https://github.com/christian-bick/edugraph-ontology/blob/db5d9541223880ba63e2eaa6f62d8db3186a9e2c/docs/content-evidence.md#L20-L47) already distinguish requested from illustrated performance; the audit identifies where an operational rubric is needed.

## 3. Operation versus strategy: a concentrated edge case

All three `Subtraction` → `SubtractionPlaceValuePartitioning` substitutions are whole-tens self-subtractions: **90−90, 70−70 and 50−50**. The images show place-value representations, but all nonzero material lies in one place and the result is zero. One worked solution explicitly says no ten is decomposed.

All six training examples carrying the strategy label were inspected. They show distinct place-value parts; one evaluates `9−6=3`, `250−230=20`, then combines `20+3=23`. Regrouping is therefore not a necessary condition. The unresolved question is whether a trivial single-place case with zero components counts as decomposing operands and coordinating partial differences.

The evidence supports a focused hypothesis that the model associates the place-value layout with the strategy label. It does not establish a general inability to distinguish subtraction from strategies. The `expands` relation licenses neither substitution nor automatic addition of the operation.

**Recommended upstream decision:** specify the required decomposition/partial-difference witness, including degenerate cases, and whether operation and strategy are both explicit. Then test matched whole-tens, nontrivial partitioning and regrouping contrasts. Do not implement a rule that “no borrowing means no partitioning.”

## 4. Other conventions and clear model targets

**Fraction specificity:** broad `FractionNumbers` is penalized instead of `ProperFractions` for 3/8 multiplication and 1/8-versus-2/8 comparison. Yet exact controls for decimal conversion of 98/100 and 7/10 retain the broad label. This supports a salience/specificity question, not a finding that broad fractions are mathematically false. In contrast, predicting `ProperFractions` for the explicit target `12/6 = 2` is clearly wrong.

**Reference frames:** all five affected images use a consistent frame. Two inch line plots receive a defensible extra `SingleFrameOfReference`; other fraction/decimal numberlines require it in gold. An exact kilogram-numberline control omits it. Define when the single-frame fact must be explicit rather than treating every such extra as hallucination.

**Operand digits:** five missed labels on four images have direct evidence: `44+2`, `38+9` and `61+4` have single-digit smaller operands; `24+30` and `61+4` have two-digit larger operands. These are ready for controlled model-improvement studies. Keep the operand rule tied to the original operation, not digits in intermediate decompositions.

**Explicit ancestors:** eight final gold sets and their exact predictions contain both `LiquidVolumes` and `VolumeMeasurement`. Two were visually inspected alongside controls that omit the ancestor. This is a serialization convention with no current error benefit from pruning; blanket pruning would break eight exact predictions. Ontology permission to omit an ancestor is not a mandate to rewrite the released set. The audit also leaves open whether conversion units alone suffice as liquid-specific evidence.

## Upstream decision package and next experiment gates

| Order | Decision to request | Acceptance evidence before model changes |
|---|---|---|
| 1 | Numeric facet applicability, task-value domain and canonical bound selection | Adjudicated examples in both substitution directions, plus frame-capacity and measurement controls; a versioned rule that independently reviewing the image can reproduce. |
| 2 | Main versus instrumental abilities; retained action evidence in solutions | Same rubric applied to explaining, reading, executing and articulating; explicit unknown-input and green-input controls; no dependence on source filenames. |
| 3 | General versus specific fractions, reference-frame inclusion, and explicit ancestors | Positive and negative emission examples, and checks of omitted as well as supplied labels. |
| 4 | Degenerate place-value strategy evidence | Decisions for the three self-subtractions and nontrivial training controls, including partitioning without regrouping. |
| 5 | Model targets that already have clear evidence | Fresh contrasts for operand digits, proper/improper fractions and explicit cognitive actions; retain broad positive controls. |

A useful change to upstream QA would be a separate **explicit-set reproducibility check** after label-support validation: reviewers derive the target under the rubric without seeing the generator's selected labels, then resolve discrepancies. A support pass should remain distinct from a completeness/specificity pass. This is a proposed procedure, not an assertion that every currently unselected true label should be added.

Any accepted changes belong in a new upstream release with provenance. Preserve the original baseline; compare changed releases separately. Avoid another training run aimed at these ambiguous targets until the corresponding rules are decided. This audit provides no measured evidence that more epochs, reward optimization or a larger model will resolve an unspecified label convention.

## Reproduction, artifacts and limitations

Baseline: `edugraph-20261002-qwen38-27b-runpod-full-a40-v1`, dataset artifact commit `9b509867e898490615be3f59bc2f31fac429389c`, ontology 0.30.0; immutable prediction bundle identified in the [full-run record](RUNPOD-FULL-A40-20261002.md).

Inspected source: dataset GitHub release tag v0.30.0-03 resolves to `d1920909e34bfe368cc22a4aea6e591690db796d`; ontology v0.30.0 resolves to `db5d9541223880ba63e2eaa6f62d8db3186a9e2c`. **Exact GitHub-source-to-Hugging-Face build provenance is unverified.** Matching release metadata is consistent, but is not a build receipt. Source mechanisms explain the inspected implementation; image claims rely on the actual training artifacts.

Local, gitignored outputs under `reports/label-convention-audit-20261003/`:

- `summary.json`: mechanically reproduced impact and coverage counts;
- `numeric-audit.json`, `abilities-strategies.json`, `fractions-operands.json`: case IDs, locators, image observations, original labels and provisional judgments;
- `upstream-policy.md`, `sources/`, `source-manifest.json`: pinned policies and implementation evidence, with immutable links and cached-source hashes;
- `evidence-manifest.json`: source/input/report hashes and all 91 verified image hashes;
- `review.html`: local gallery of unmodified images, notes and collapsible original gold/predictions.

Reproduce measurements and the gallery with the existing local artifacts:

```powershell
uv run --no-sync python temp/label-convention-audit/summarize.py
```

The script checks input identities and baseline counts; it does not automate the visual judgments. Scratch tools and generated evidence remain local, not published or committed dataset copies. This document preserves the audit protocol, findings and decision requests in project documentation. No gold-change recommendations are treated as approved.
