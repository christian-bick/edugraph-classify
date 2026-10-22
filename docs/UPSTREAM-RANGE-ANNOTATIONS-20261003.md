# Proposal: consistent numeric range annotations

Prepared 2026-10-03 for review in `edugraph-dataset` / the local `edugraph-content` checkout. This is an analysis and proposed implementation contract, not a change to published labels, ontology semantics, or classifier evaluation.

## 1. Decision and evidence boundaries

The user has authorized an upstream-ready analysis of consistent numeric ranges and legitimate observable label coverage. The proposed **strongest supported range per rendered task** rule below still needs upstream acceptance. It is more specific than the existing requirement that an emitted range be true.

Three source identities must remain separate:

| Source | Immutable revision | Use here |
| --- | --- | --- |
| Dataset v0.30.0-03 source | `d1920909e34bfe368cc22a4aea6e591690db796d` | Explains the release-source generator and annotation mechanisms |
| Current dataset development HEAD | `374055902a5105d39408ba43e72c3dbea56ce7e5` | Locates implementation work; tracked working tree was clean when inspected |
| Ontology v0.30.0 | `db5d9541223880ba63e2eaa6f62d8db3186a9e2c` | Meaning of the baseline's labels and relations |

The trained Hugging Face dataset revision is `9b509867e898490615be3f59bc2f31fac429389c`. The preceding audit did not establish an immutable build receipt connecting that artifact to the same-named GitHub release source. Therefore source behavior supports a mechanism diagnosis; it does not prove that replaying that commit reproduces every released pixel. Current local generated output is development output, not evidence of release provenance. No upstream files or artifacts were changed during this analysis.

The [image audit](LABEL-CONVENTION-AUDIT-20261003.md) found 18 upper-bound substitutions for which both competing bounds were true under the recorded quantity interpretation. Examples include `6 + 2 = 8` with gold `NumbersSmaller10` and prediction `NumbersSmaller20`, and `2 × 6 = 12` with gold `NumbersSmaller100` and prediction `NumbersSmaller20`. This is a specificity/inclusion problem in those cases, not evidence that all disputed numeric labels were false or that the model cannot count. The observed sample is not an independent estimate of population annotation quality.

## 2. What the current contract already establishes

**Ranges describe absolute values of task inputs and results, inclusively.** They include an unshown answer and a small adjustment operand. The upstream September 27 clarification explicitly treats `342 − 10 = 332` as supporting a lower bound of 10, not 100. It distinguishes actual operands from digits and place-value coefficients used only to represent a quantity. These are already recorded decisions, not new recommendations. [Release clarification, lines 12–38](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/docs/plan/numeric-range-bounds.md#L12-L38), [ontology range definitions, lines 721–850](https://github.com/christian-bick/edugraph-ontology/blob/db5d9541223880ba63e2eaa6f62d8db3186a9e2c/core-scopes-math.ttl#L721-L850).

**The implementation resolves generation constraints, not instance extrema.** `resolveRangeFromLabels` maps selected labels to `min`/`max` and uses an exact target-context selection. `planned-generation.ts` combines invariant and resolved schema labels before calling the generator. A sample drawn from a `≤100` configuration can therefore use only values below 20 while retaining `≤100`. Changing threshold ordering in the resolver would not solve that distinction. The relevant files are unchanged between the inspected release and development revisions. [Resolver, lines 80–149](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/src/lib/ontology.ts#L80-L149), [annotation assembly and draw, lines 62–93](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/src/lib/planned-generation.ts#L62-L93).

**Several families already enforce useful quantity boundaries.** The offset helper checks the fixed step against the range, then bounds the starting value and answer. The two-step word-problem helper explicitly checks operands, intermediate, and final answer. Count-out generation bounds both the requested count and the available collection. These are reusable precedents, not evidence that every family has a complete extractor. [Offsets, lines 11–37](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/src/generators/counting/counting-offset.ts#L11-L37), [two-step values, lines 5–8 and 36–56](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/src/generators/arithmetic/word-problem-evidence.ts#L5-L56), [count-out, lines 12–21](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/src/generators/counting/counting-selection/generator.ts#L12-L21).

**Accessible evidence remains the limit.** Necessary mathematical consequences may count; hidden author intent may not. The canonical payload is a source of evidence only insofar as the task exposes or entails the relevant quantity. [ONT-E1, lines 7–18](https://github.com/christian-bick/edugraph-ontology/blob/db5d9541223880ba63e2eaa6f62d8db3186a9e2c/docs/content-evidence.md#L7-L18).

## 3. Proposed quantity domain

Define a versioned, typed `rangeEvidence` receipt for the **actual task projection**, separately from the generator's allowed interval. It records included values, roles, units where relevant, source payload paths, and whether the value is visible or necessarily implied. It also records excluded representational quantities with a reason. Do not recursively collect every numeric payload field or OCR token.

For an auditable first version:

| Task family | Include as range inputs/results | Exclude unless independently requested or used mathematically |
| --- | --- | --- |
| Arithmetic and inversion | All operands and final result, including a uniquely determined blank | Operand-position indexes, rendered digit glyphs, layout sizes |
| Explicit multi-step task or worked derivation | Inputs/results of each required or actually illustrated step | Intermediate values invented by an optional solution algorithm |
| Counting and comparison | Each quantity being counted/compared; requested and available counts for count-out | Sum of unrelated compared groups; unused frame capacity |
| Place-value reading | Whole represented quantity | Coefficients/digits used solely to encode that quantity, such as the 7 in “7 tens” |
| Fraction value task | Exact rational values serving as inputs/results | Numerator/denominator as separate integer magnitudes when merely notation |
| Fraction-construction or numerator/denominator transformation task | The rational values plus independently requested counts or arithmetic operands/results on numerator/denominator | Unused subdivision indexes and pixel coordinates |
| Measurement calculation | Numerical magnitudes in the units the task actually uses, plus count/multiplier operands and results | Automatic conversion into an arbitrary canonical unit; formatting denominators such as 100 used to store cents |
| Number line, graph, or frame | Values of task-relevant points/data and required outputs; scale/capacity values when they are themselves requested or operands | Incidental tick labels, unused endpoints, empty slots, and drawing bounds |

The fraction and measurement distinctions require view-aware tests. For example, the fraction-number-line producer stores numerator, denominator, whole-count, and step-index data; those fields do not all denote independent arithmetic inputs. Measurement problems already use rational numerator/denominator storage for integer, decimal, and fractional quantities, making a numeric-field walk especially unsafe. [Fraction payload, lines 42–62](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/src/generators/fraction/fraction-number-line/generator.ts#L42-L62), [measurement values and sampling, lines 18–44 and 72–100](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/src/generators/measurement/measurement-word-problems/generator.ts#L18-L100).

For multi-step tasks, include a required intermediate even if unshown. For a free-method arithmetic question, do not add every intermediate that some valid method could use. A worked solution that explicitly introduces an additional arithmetic intermediate can consequently have a different range from its question. Keep the underlying task family in one split, but adjudicate each classification image's evidence independently. The existing triple sampler's optional `boundIntermediateProducts` guard illustrates why “multiply by zero” must not hide a large intermediate that a property view actually displays. [Sampler, lines 111–123](https://github.com/christian-bick/edugraph-dataset/blob/d1920909e34bfe368cc22a4aea6e591690db796d/src/generators/arithmetic/arithmetic-triple-sampling.ts#L111-L123).

For underdetermined tasks, use only a finite, justified value domain or a proved bound over all admissible task values. If neither is available, mark the range evidence unresolved instead of taking the hidden generated answer as truth. Report unsupported family coverage explicitly; absence of an extractor must not silently mean no numeric range applies.

**Do not reuse this domain blindly for other facets.** `NumbersWithoutZero` and `NumbersWithoutNegatives` have their own definitions. ONT-E4 explicitly notes that a displayed scale can include zero; an incidental axis excluded from the input/result range may still affect a zero-presence claim. A zero digit in `10` is not a zero-valued quantity. Any joint audit needs facet-specific domains and an explicit resolution for such cases. [ONT-E4, lines 49–65](https://github.com/christian-bick/edugraph-ontology/blob/db5d9541223880ba63e2eaa6f62d8db3186a9e2c/docs/content-evidence.md#L49-L65), [sign and zero definitions, lines 851–867](https://github.com/christian-bick/edugraph-ontology/blob/db5d9541223880ba63e2eaa6f62d8db3186a9e2c/core-scopes-math.ttl#L851-L867).

## 4. Proposed strongest-bound rule and its gates

For a nonempty, established range domain `Q`, compute `m = min(abs(q))` and `M = max(abs(q))` using exact rational arithmetic where necessary. The v0.30.0 positive thresholds are `5, 10, 20, 100, 120, 1000, 10000, 100000, 1000000`.

1. Emit the upper-bound label with the smallest available threshold `U` for which `M ≤ U`.
2. Emit the lower-bound label with the largest available positive threshold `L` for which `L ≤ m`.
3. Omit looser bounds from the explicit range serialization; keep their justified implication closure separate. This is an application policy, not `specializes` ancestor pruning.
4. If there is no fitting upper threshold, emit no upper bound and report “outside catalog”; never fabricate a label or clamp the quantity. If `m < 5`, no positive lower bound applies.
5. Decide explicitly whether to serialize `NumbersLargerZero`. The proposed compact policy omits this universal absolute-value floor, but that is an information-policy decision, not a claim that zero is absent. The ontology says `-3`, `0`, and `5` all satisfy it. A complete justified-facts view would still need to recover this fact from the confirmed nonempty numeric domain; plain ontology closure from an upper bound alone is not guaranteed to recover it. Record that evidence rule, or emit the floor uniformly. Mixing conventions by family is the problem to avoid.

Equality requires an upstream semantic/API decision before release. `Q={10}` supports both `NumbersLarger10` and `NumbersSmaller10` by their inclusive definitions. The ontology intentionally records a **partial descriptor-level conflict**, and its documented helpers honor that exclusion rather than computing an exact interval intersection. This is already true in the pinned ontology; it is not a newly discovered erroneous edge. Preserve inclusive meanings, document a separate instance-level numeric consistency check, and decide how annotation validation and generation matching consume it. Do not arbitrarily drop one true bound, turn equality into a strict inequality, or globally reinterpret every `contradicts` edge. [ONT-R2, lines 58–82](https://github.com/christian-bick/edugraph-ontology/blob/db5d9541223880ba63e2eaa6f62d8db3186a9e2c/docs/relations.md#L58-L82).

Two other gates apply:

- **Context gate:** range annotations should describe a meaningful numeric task. The fact that a geometric diagram contains a page number does not make it a numeric-range task. The inclusion policy must name supported contexts, rather than manufacture bounds for every visible numeral.
- **Evidence gate:** mathematical truth alone does not establish which values belong to the task domain. Capacity, fractions, units, and unknowns must pass the role rules above before extrema are computed.

## 5. Architecture options and recommended sequence

The source architecture explicitly describes a constraint satisfier, with labels resolved outside generators before the mathematical draw. Implementing strongest **realized** bounds changes that contract; it is not a small patch to `resolveRangeFromLabels`. [Current IMPL-G1–G4, lines 11–49](https://github.com/christian-bick/edugraph-dataset/blob/374055902a5105d39408ba43e72c3dbea56ce7e5/docs/implementation-generator.md#L11-L49).

**Recommended: first build a shadow audit, then consider a separate annotation layer.** Preserve `targetLabels`, generation-capability labels, selected configuration, plan and replay receipts. Add typed mathematical evidence extractors and view-projection evidence outside pure generators/rendering. A shared annotator computes proposed realized bounds and writes an explicit delta report; it changes no gold in this phase. Record the annotation policy and ontology identity in every receipt.

If accepted, retain those generation constraints and proofs while emitting the separately versioned observable annotation set. For a requested `≤100` task realized as `2 × 6 = 12`, preserve the original generation request and prove that the realized values satisfy it; explicit annotation may then use `≤20` under the new policy. Current positive matching and `assertResolvedTargetCoverage` use equality/specialization capability coverage, not general `implies` closure. Replacing `≤100` in that machinery with `≤20` will not automatically preserve target coverage. Introduce a specifically documented annotation-versus-constraint proof and target association receipt; do not broaden generic matching to all implications to make the change pass. Keep render configuration based on the accepted plan, not the newly computed labels.

**Alternative: sample only instances whose strongest bounds equal selected bounds.** This preserves pre-draw labeling more directly, but turns ranges into exact buckets, changes distributions, increases retries, and can remove valid target/view matches. The lower and upper extremes must both witness the bucket. This should be evaluated as a deliberate generation redesign, not an invisible filter applied to historical samples.

Neither option licenses classifier-side relabeling of the released baseline. Publish any accepted annotation revision upstream with a migration manifest, then pin it in a new classifier experiment. Keep raw baseline labels and predictions for historical comparisons.

## 6. Concrete work items

| Work item | Implementation locations / checks | Completion evidence |
| --- | --- | --- |
| Ratify quantity and serialization policy | `docs/plan/numeric-range-bounds.md`, `docs/label-architecture.md`, `docs/spec-general.md`, `docs/implementation-generator.md`; ontology numeric definitions and ONT-R2 integration | Decisions for equality, zero floor, units, intermediates, capacities, axes, and underdetermined tasks |
| Build pure quantity evidence extraction | New shared range-evidence module; family adapters around arithmetic helpers, `counting-offset.ts`, `counting-selection`, fractions and measurement payloads; task projection adapter contracts | Typed roles; exact values; visible/necessary witness; exclusions; explicit unsupported status |
| Keep constraint resolution separate | Existing `src/lib/ontology.ts` and `src/lib/planned-generation.ts`; `src/lib/label-contracts.ts` | Existing generation configuration unchanged by annotation audit; separate constraint-satisfaction proof for narrower realized bounds |
| Prototype audit and receipts | `src/lib/dataset-metadata.ts` stores plan/replay and target associations already; extend operational metadata, not public rows by hand | Replayed artifact identity, source pin, policy version, extrema, proposed bounds, old/new delta and review outcome |
| Integrate approved annotation policy | Dataset creation/merge, VQA inputs, asset-index target association checks, dependency identities | Policy changes invalidate annotation/VQA/coverage records; label-only changes do not pretend pixels changed; aliases retain target provenance |
| Test and release | Unit tests beside helpers and adapters; catalog/integration suites; representative canonical rendering and image review | Acceptance matrix passes; deterministic full-release audit; no new image conflicts or split leakage; new upstream release |

Use upstream `npm run test`, `npm run test:integration`, `npm run test:coverage`, and relevant type/spec checks when implementation is authorized. No code tests or renders were run for this documentation-only proposal. Live VQA and generation/publication remain separate actions with their own authorization and cost boundaries.

The first audit should report per family/view: applicability coverage; unsupported or ambiguous domains; true-but-noncanonical bounds; false existing claims; proposed additions/removals; equality-policy cases; target-proof failures; and Q/S differences. Freeze the reviewed policy before measuring a new classifier. Relabeling based on model errors alone would bias the evaluation.

## 7. Acceptance matrix for the proposed policy

These are test specifications, not an assertion that all fixtures exist or all decisions are accepted. `≥` and `≤` below are inclusive bounds on absolute value; a dash means no positive lower bound. All results are conditional on the quantity-domain decisions above.

| Fixture | Range domain / key exclusion | Expected strongest bounds or review result |
| --- | --- | --- |
| `1 + 3 = □` and its solution | `{1,3,4}`; include hidden answer | `—, ≤5`; same range in both modes |
| `2 × 6 = 12` | `{2,6,12}` | `—, ≤20`, regardless of generator's `≤100` allowance |
| `342 − 10 = 332` | `{342,10,332}` | `≥10, ≤1000`; never `≥100` |
| `80 − 75 = □` | `{80,75,5}` | `≥5, ≤100`; operands alone must not yield `≥20` |
| `10 − 10 = 0` | `{10,10,0}` | `—, ≤10`; separately verify zero facet |
| `−12 + 7 = −5` | `{−12,7,−5}` | `≥5, ≤20`; sign does not change magnitude test |
| Count/compare only the value 10 | `{10}` | `≥10, ≤10` mathematically; explicit endpoint/API policy required |
| Count seven objects in a 20-slot frame | `{7}`; unused capacity excluded | `≥5, ≤10`; capacity alone must not force `≤20` |
| Fill the same frame to 20 | `{7,13,20}` if the requested task is finding missing objects | `≥5, ≤20`; task role changes the domain |
| Compare two groups of 8 | `{8,8}`; do not sum unrelated groups | `≥5, ≤10` |
| Read “7 tens” as 70 | `{70}` if coefficient is solely representational | `≥20, ≤100`; retain a role witness for excluding 7 |
| Locate `5/4` on a number line | Rational `5/4`; unused tick indexes excluded | `—, ≤5` under the proposed fraction-value policy; scale/zero semantics separately reviewed |
| Scale numerator/denominator by 10 | Independently operated-on integers and rational values | Must include the actual transformation operands/results; cannot reuse the locate-only extractor |
| `18 m − 13 m = 5 m` | `{18,13,5}` in the displayed unit | `≥5, ≤20`; internal unit conversion must not alter annotation |
| Explicit `1 m = 100 cm` conversion | `{1,100}` plus any independently used operands | `—, ≤100`; unit conversion policy recorded |
| Explicit `(9 × 9) × 0 = 0` derivation | Include intermediate 81 | `—, ≤100`; do not infer `≤10` from final zero |
| Numeric input outside largest upper threshold | Include actual finite value | No catalog upper label; report out-of-catalog without clamping |
| Unresolved blank with several possible values | Domain not uniquely established | Bound only with a proof over all admissible values; otherwise unresolved |
| Shape-only task with incidental page number | No applicable task quantity domain | No numeric-range addition |

Also test exact thresholds and adjacent rational values, zero normalization, non-finite inputs, empty domains, operand permutations, stable serialization, and equivalent rational storage (`1/2` versus `50/100`). Validate that a purely presentational change leaves annotations unchanged, while an actually different mathematical task can change them.
