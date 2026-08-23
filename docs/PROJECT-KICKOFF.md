# EduGraph VLM classifier: project kickoff and baseline decision

Status: accepted starting architecture  
Last provider review: 2026-08-23  
Initial provider: Fireworks AI

## 1. Executive decision

Build the first credible classifier as a managed, direct-labeling VLM baseline on Fireworks AI before investing in a custom training and serving stack.

The first system is deliberately small:

```text
one isolated pedagogical task image
                 |
                 v
       fine-tuned open-weight VLM
                 |
                 v
       schema-constrained label JSON
                 |
                 v
     deterministic validation/canonicalization
                 |
                 v
        explicit ontology label set
```

The baseline excludes retrieval, an ontology-aware neural decoder, candidate reranking, a custom multilabel head, and trained reasoning traces. Those remain evidence-driven follow-ups. The managed baseline establishes whether they solve a real residual problem and gives every later design a quality, latency, cost, and engineering-complexity reference.

Fireworks is preferred over Together for this phase because it currently offers a coherent progression from managed vision SFT to preference/RL recipes and a Training API with custom Python objectives. This does **not** mean that managed SFT exposes a set-level multilabel loss: its standard objective remains response-token cross-entropy. It means the provider leaves a plausible next step without requiring immediate self-managed GPU infrastructure.

## 2. Problem contract

### 2.1 Unit of analysis

The base assumption is:

> One independent pedagogical task equals one classification unit.

Worksheet regions, document sections, or video segments are isolated upstream. The classifier does not have to discover task boundaries while labeling. Question and worked-solution renderings may both be valid samples, but every underlying task and all of its renderings must remain in the same dataset split.

This isolation-first design reduces ambiguity, improves error attribution, and lets small VLMs spend capacity on visual/mathematical evidence rather than segmentation.

### 2.2 Objective and terminology

The original quality requirements translate to standard multilabel terminology as follows:

- **Precision:** every predicted label is correct; minimize extra labels.
- **Recall:** every gold label is present; minimize missing labels.
- **Exact-set match** (also called subset accuracy): the predicted set equals the gold set. This is the primary metric because it requires precision and recall simultaneously.
- **Parameter count:** the requested 4B-35B range describes model size. Context length is separately measured in tokens.

“Accuracy” alone is ambiguous for multilabel classification and should not be used without naming the exact metric.

### 2.3 Explicit-label contract

The model predicts the minimal explicit concepts that the task independently demonstrates. It does not repeat knowledge that can be recovered mechanically from the pinned ontology.

For example, if a specific solving concept is explicitly demonstrated and its ancestors are connected by `partOf`, the output contains the specific concept, not the entire ancestor chain. Consumers that need a broader view may compute a derived closure later. Explicit predictions and derived labels must remain distinguishable in storage and evaluation.

The recommended response shape is dimension-aware and deterministic:

```json
{
  "areas": ["<area-id>"],
  "scopes": ["<scope-id>"],
  "abilities": ["<ability-id>"]
}
```

Properties and labels use one fixed canonical order. Arrays are sets semantically, but canonical ordering removes multiple textual encodings of the same target during generative SFT.

## 3. Repository boundaries

### `edugraph-dataset`

Owns released labeled images, official train/validation splits, public metadata, and the evidence contract for each image's complete label conjunction.

### `edugraph-ontology`

Owns label identifiers, dimensions, definitions, versions, and semantic relations.

### `edugraph-classify`

Owns:

- provider-neutral prompt contracts and versioned Fireworks prompt variants;
- deterministic conversion of released EduGraph data to provider JSONL;
- ontology pinning, output schemas, and canonicalization;
- Fireworks dataset/job/deployment API orchestration;
- base-model, prior-checkpoint, and fine-tuned-model evaluation;
- run manifests, raw predictions, aggregate metrics, and error analysis;
- later DPO/RFT data construction if baseline errors justify it.

It must not duplicate released images, large provider upload files, model weights, credentials, or generated reports in Git. Generated artifacts should be reproducible from a dataset release plus an ontology version.

## 4. Current data reality

The local union dataset inspected during this design had 1,958 images: 1,602 train images and 356 validation images, evenly divided between question and solution renderings. It contained 295 distinct labels and 659 exact label sets, with 2-12 labels per sample and most samples carrying 4-10 labels.

The important constraint is not the number of outputs but the long tail:

- all 295 labels appeared in train, but only 191 appeared in validation;
- 104 train labels were therefore not measurable per-label on validation;
- 135 labels occurred at most 9 times in train;
- 42 labels occurred at most twice in train;
- 183 labels occurred fewer than 20 times in train.

These figures are a dated snapshot, not constants. The new repository must regenerate and store a data-profile report for every pinned dataset release.

Consequences:

1. Overall micro metrics can conceal rare-label failure.
2. Exact-set performance can be limited by sparse label combinations even when visual recognition is good.
3. The first run should preserve the natural distribution. Oversampling and sample weights are experiments, not silent preprocessing defaults.
4. A future held-out test set must cover the production-critical labels and combinations; the present validation split cannot measure every label.

## 5. Ontology semantics and canonicalization

Canonicalization is a deterministic boundary between raw model output and final explicit labels. It is not a second semantic classifier and must never rescue arbitrary model guesses.

Minimum rules:

1. Reject or record any identifier absent from the exact pinned ontology.
2. Validate that every identifier occurs in the declared Area, Scope, or Ability field.
3. Remove duplicate identifiers and impose canonical order.
4. When both a concept and its `partOf` ancestor are predicted, retain the most specific explicit concept.
5. Reject logically contradictory label sets using `contradicts`.
6. Keep raw and canonical predictions so improvements from schema enforcement and graph cleanup are measurable rather than hidden.

Relation meanings must not be conflated:

- `partOf` supports taxonomy closure and ancestor redundancy removal.
- `implies` represents logical entailment and may support a separate derived view.
- `contradicts` supports hard exclusion.
- `integrates` describes composition/progression: A is synthesized using B. It is not logical entailment.
- `expands`, `inverts`, and `translates` are useful progression signals and later candidate/reranking features, not automatic output rules.

In particular, `A integrates B` does not by itself mean “always add B” or “always remove B.” If B is merely instrumental while performing A, the explicit-output policy normally omits B. If the isolated task independently elicits B, B remains valid. The atomic-task boundary makes that evidence decision substantially cleaner.

## 6. Area, Scope, and Ability correlation

Areas, Scopes, and Abilities are strongly correlated and the baseline should expose that structure without turning correlations into brittle rules.

The grouped output contract helps the autoregressive model learn a joint set and allows dimension-specific evaluation. It also prevents a label from silently occupying the wrong semantic role. However:

- ontology facts are hard constraints;
- implementation compatibility is an authored constraint;
- observed co-occurrence is a statistical prior.

Those are different kinds of information. Abilities are intentionally reusable across mathematical domains, so a currently rare Area-Ability pairing must not become permanently impossible merely because it was absent from one dataset release.

The direct VLM learns cross-dimension compatibility implicitly from joint examples. A later custom model, if justified, should use soft bidirectional refinement or joint set scoring rather than a strict Area -> Scope -> Ability cascade, which would propagate early mistakes.

## 7. Why a generative VLM baseline first

A generative VLM can already:

- read the task image;
- use mathematical context to disambiguate visually similar exercises;
- emit several correlated dimensions jointly;
- produce only a short canonical response;
- accept new label identifiers without changing a neural output layer;
- run behind Fireworks' managed training and deployment APIs.

At 295 or even roughly 1,000 labels, raw output dimensionality is not itself a compute problem. The decisive questions are whether the model learns rare concepts, avoids plausible-but-unjustified labels, respects the minimal-explicit-label contract, and generalizes across task renderings.

Starting here also keeps the first causal comparison interpretable: base model versus the same model after direct SFT, under one prompt, one output schema, one dataset split, and one inference configuration.

## 8. Fireworks capability boundary

The following statements were checked against official Fireworks material on 2026-08-23 and must be rechecked before each launch:

- Managed VLM SFT accepts OpenAI-style JSONL with multimodal `content` and base64 data-URI images.
- Managed SFT trains desired responses with the standard supervised token objective; the Training API cookbook describes response-token cross-entropy with prompt tokens masked.
- Managed jobs support a separate evaluation dataset, LoRA rank, epochs, learning rate, sample-count batch size, and maximum context length.
- Fireworks structured outputs can enforce a JSON Schema at inference.
- The Training API exposes custom Python losses over differentiable token log-probabilities and supports VLM-compatible recipes/shapes, but that is a later, more manual path.
- Fine-tuned LoRA serving requires a dedicated/on-demand deployment rather than a serverless endpoint.

Model eligibility is volatile and model-specific. The live Fireworks catalog currently marks `qwen3-vl-30b-a3b-instruct` as fine-tunable, while the 8B Qwen3-VL page says fine-tuning is not supported. Older Qwen2.5-VL documentation announced 3B, 7B, and 32B training support, while current individual catalog cards are not fully consistent. Therefore every run begins with:

```bash
firectl model get -a fireworks <MODEL_ID>
```

and proceeds only when the live result reports `Tunable: true` and a compatible vision training shape/method.

The practical model ladder is:

1. Evaluate the existing Qwen3-VL-4B EduGraph checkpoint as historical reference where it can be served reproducibly.
2. Fine-tune the smallest currently tunable Qwen VLM near the 4B-8B range.
3. Compare with Qwen3-VL-30B-A3B Instruct if it remains tunable. It has roughly 31B total parameters and 3B active parameters, so it fits the stated total-parameter ceiling while offering an informative MoE comparison.
4. Add a different open-weight VLM family only if Fireworks marks it tunable and the Qwen results leave a useful family-level question.

Model availability is a preflight result captured in the run manifest, not a source-code assumption.

Managed Fireworks documentation does not expose a clear switch for freezing or unfreezing the VLM vision encoder. The baseline must record the actual provider training configuration and must not claim the vision tower is frozen unless the live job surface proves it.

## 9. Baseline training specification

### 9.1 Training datum

Each source row becomes one single-turn conversation:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<versioned direct-labeling contract>"
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Classify this independent pedagogical task."},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/png;base64,<encoded image>"}
        }
      ]
    },
    {
      "role": "assistant",
      "content": "{\"areas\":[...],\"scopes\":[...],\"abilities\":[...]}"
    }
  ]
}
```

The prompt should be short and stable. It should say that the input is one isolated task; demand only explicit, independently evidenced, most-specific labels; forbid ancestors, merely instrumental competencies, and prose; and define the three output dimensions. It should not paste all ontology definitions or the graph into every sample.

Images are encoded only in generated provider-upload artifacts. The converter validates MIME type, row count, hashes, image existence, split integrity, ontology membership, dimensions, and exact output serialization before upload.

### 9.2 Objective

Use response-only SFT. System/user/image tokens are context; assistant JSON tokens receive loss. Fireworks managed SFT supplies token-level cross-entropy rather than a bespoke set loss. For the baseline this is acceptable because:

- the target is extremely short;
- there is exactly one canonical textual representation per label set;
- schema-constrained inference eliminates format and vocabulary drift;
- the real set objective is measured explicitly during evaluation.

The mismatch must remain visible: lower validation token loss does not necessarily imply better exact-set match.

Use uniform sample weight for the first experiment. Later sample weighting or rare-label oversampling must be an explicit, versioned experiment because it can distort label cardinality and cross-dimension co-occurrence.

### 9.3 Initial job policy

- Tuning: LoRA.
- LoRA rank: provider default first; compare 16 or 32 only if diagnostics indicate underfitting. Current managed documentation caps rank at 32.
- Epochs: one initial pass; add a second run only if training/evaluation evidence indicates underfitting. Managed early stopping is unavailable.
- Learning rate and scheduler: provider defaults first.
- Batch size: provider default sample-count batch unless a measured reason requires change.
- Context: request 4,096 tokens if Fireworks' rendered samples prove that all image/text/answer datums fit; otherwise use the smallest safe value such as 8,192. Never allow silent truncation.
- Training split: released train only.
- Evaluation split: upload the released validation split explicitly; do not allow an automatic carve-out.
- Secrets: environment or secret manager only; never manifests, logs, prompts, or Git.

Before a paid job, inspect Fireworks Render Samples to verify the processor, image tokens, chat template, assistant response, and non-zero response loss mask.

## 10. Inference contract

Use the identical versioned system prompt and image-preparation policy used for training. Set deterministic decoding (`temperature: 0`) and a small output-token cap.

Request a JSON Schema that:

- requires exactly `areas`, `scopes`, and `abilities`;
- allows arrays only;
- restricts items to identifiers from the pinned ontology and correct dimension;
- disallows duplicate items and additional properties where supported.

The schema is an inference constraint, not evidence that a chosen label is semantically correct. Record both the model's schema-constrained raw set and the post-canonicalization set.

Fireworks currently documents that `response_format: json_schema` disables reasoning output on reasoning-capable models. That is desirable for the direct baseline and another reason to test reasoning separately.

## 11. Evaluation and analysis

### 11.1 Required metrics

Primary:

- exact-set match over all explicit labels.

Required diagnostics:

- micro precision, recall, and F1;
- macro precision, recall, and F1 over labels with evaluation support;
- exact match and micro/macro metrics per Area, Scope, and Ability;
- question-only versus solution-view performance;
- frequent, medium, rare, and zero-validation-support label coverage;
- label-set cardinality error;
- missing-label and extra-label counts;
- invalid JSON, unknown identifiers, wrong-dimension labels, duplicates, redundant ancestors, and contradictions;
- raw versus post-canonicalization metrics;
- latency, input/output tokens, deployment configuration, and estimated cost.

For every aggregate score, report the number of examples, labels, and exact label sets it actually covers.

### 11.2 Comparisons

The first report should compare, under the same evaluator and output contract where possible:

1. an eligible untuned base VLM;
2. the existing Qwen3-VL-4B EduGraph checkpoint;
3. the new Fireworks SFT model;
4. optionally, a larger tunable Qwen VLM if the first managed model shows a useful scaling trend.

Do not promote a model based on validation loss or F1 alone. It must improve exact-set match and show an acceptable precision/recall balance on both question and solution views.

### 11.3 Error taxonomy

Every incorrect example should support automatic attribution to one or more buckets:

- format/schema failure;
- visually missed evidence;
- missing label;
- extra plausible sibling;
- redundant ancestor;
- unjustified `integrates`-neighbor label;
- contradiction;
- wrong dimension;
- problem-only reasoning failure;
- rare/unseen label;
- label-definition or gold-data concern.

This taxonomy decides the next intervention. It prevents adding architectural complexity to solve what is actually a prompt, data-coverage, or canonicalization defect.

## 12. Reasoning: role and deferred experiments

Reasoning is most useful when only the unsolved problem is visible and the label depends on an operation, procedure, or constraint that must be inferred by solving or mentally simulating the task. It is less useful for visually explicit classification and can reduce precision by surfacing concepts that are related but not independently demonstrated.

Reasoning can be tested without fine-tuning by comparing an instruct/direct mode with a thinking-capable model under the same images and scoring only the final label set. That experiment should be split by question versus solution views.

Do not include free-form chain-of-thought in the first SFT. If direct SFT errors concentrate in the question-only reasoning bucket, a later reasoning-tuning dataset should contain short, verifiable supervision derived from generator truth, for example:

- visible observations;
- necessary mathematical derivation;
- evidence mapped to each retained label;
- final canonical JSON kept separate from the reasoning trace.

Fireworks supports `reasoning_content` for compatible managed models, but a usable trace still has to be correct. Unverified synthetic prose is more likely to teach rationalization than reliable classification.

## 13. Evidence-driven follow-up ladder

Follow-ups are selected from measured baseline failures:

1. **Format errors:** strengthen schema/grammar and serialization validation.
2. **Rare-label recall:** add coverage, carefully controlled resampling, or sample weights.
3. **Extra/missing near-neighbor labels:** generate hard preference pairs and evaluate DPO/ORPO.
4. **Exact-set reward mismatch:** evaluate RFT/GRPO with a deterministic reward, such as a dominant exact-match term plus a smaller F1 term and hard rejection of invalid output.
5. **Problem-only failures:** test inference reasoning, then verified reasoning SFT.
6. **Incoherent cross-dimension combinations:** add graph-aware reranking or joint selection.
7. **Persistent generative ceiling:** evaluate a custom fixed multilabel architecture on self-managed infrastructure.

Hard negative pairs can be generated deterministically from gold sets:

- remove one required label;
- add a sibling;
- add a redundant ancestor;
- add an unjustified `integrates` neighbor;
- add a contradiction or wrong-dimension label.

This is more targeted than teaching arbitrary bad outputs.

## 14. What a custom 295-1,000-label head would change

A fixed multilabel head is computationally cheap. With a 4,096-wide representation and 1,000 labels, the final linear layer has about 4.1 million weights, roughly 0.1% of a 4B model. Its benefits would be direct positive/negative supervision, per-label scores and calibration, no label-name hallucinations, and fast joint graph selection.

Its real difficulties are not matrix multiplication:

- severe label imbalance;
- a fixed vocabulary and migration problem as the ontology grows;
- independent sigmoid outputs do not automatically learn Area-Scope-Ability consistency;
- very small false-positive rates accumulate across hundreds of negative labels.

For illustration, with 8 true labels among 1,000 candidates and a 0.1% false-positive rate per negative label, the probability of producing no false positive is only about `(0.999)^992`, or 37%. At 0.01% it is about 91%. Exact-set quality therefore depends heavily on calibration and structured decoding.

Hard prefiltering is not needed merely to score 1,000 labels. It can also impose a recall ceiling: retaining each of 8 gold labels with 99% probability retains all 8 only about 92% of the time. Candidate filtering is justified only before an expensive definition/evidence verifier and must be designed for extremely high recall, with per-dimension candidates, graph neighbors, and an escape route.

If a custom architecture becomes necessary, the preferred direction is:

- a shared VLM representation;
- separate Area, Scope, and Ability scores;
- soft bidirectional cross-dimension refinement;
- a graph-constrained joint set decoder;
- label embeddings built from definition, dimension, and graph context, plus a learned per-label residual.

That design can grow with the ontology better than a wholly unrelated learned vector per label. It likely requires a custom Transformers/GCP training and serving pipeline. Fireworks' custom loss API operates on token log-probabilities; it should not be assumed to support adding and serving an arbitrary new 1,000-sigmoid architecture through managed SFT.

## 15. Proposed repository shape

Implementation should stay provider-aware at the edge and provider-neutral in data/evaluation logic:

```text
edugraph-classify/
  .python-version
  README.md
  pyproject.toml
  uv.lock
  docs/
    PROJECT-KICKOFF.md
  prompts/
    direct-labeling-v1.md
  schemas/
    prediction.schema.json
  src/edugraph_classify/
    __init__.py
    data/          # release loading, validation, provider JSONL rendering
    ontology/      # pin loading, dimensions, canonicalization, graph checks
    providers/     # Fireworks API adapter and live capability preflight
    inference/     # request construction and raw response capture
    evaluation/    # set metrics, slices, comparisons, reports
    manifests/     # immutable experiment identity
    cli.py
  tests/
  .env.example
```

The initial scaffold uses Python 3.12 and `uv` as its project manager and native build backend. It directly depends on the official `fireworks-ai` SDK and Hugging Face `datasets` library, with all transitive versions recorded in `uv.lock`. Future implementation defaults remain typed configuration, Pydantic schemas, a small CLI, and pytest. Provider calls should be thin adapters so offline data validation and evaluation do not require Fireworks credentials.

## 16. Reproducibility contract

Every training or evaluation run gets an immutable manifest containing at least:

- EduGraph dataset repository, release tag, split, and content hashes;
- ontology package/version and graph/schema hash;
- code commit;
- prompt and response-schema hashes;
- provider, base model ID, live tunability result, training shape, and processor/renderer identity when available;
- Fireworks dataset, job, output-model, and deployment IDs;
- all hyperparameters and provider defaults resolved to concrete values;
- data-profile summary and conversion counts;
- inference configuration;
- timestamps, status, and output artifact hashes.

The API orchestration must be idempotent where possible. A dry-run validates and profiles data without uploading or spending money. Launch and deployment are explicit commands. Evaluation should be resumable and cache raw provider responses by complete request identity.

## 17. Baseline completion criteria

The managed baseline is complete when:

- data conversion is deterministic and all source/gold labels validate against the pinned ontology;
- split grouping prevents task/rendering leakage;
- a live capability preflight records a tunable Fireworks VLM;
- Fireworks Render Samples prove correct image processing and response-only loss masks;
- base, historical, and SFT predictions are evaluated with the same metric implementation;
- structured inference produces zero format and unknown-label failures;
- raw and canonical exact-set, precision, recall, per-dimension, per-view, and frequency-slice results are published as a reproducible report;
- the report recommends the next intervention from observed error buckets, including “none” if the direct baseline is already sufficient.

The eventual production threshold should be chosen after the first honest baseline exposes the achievable distribution. It should not be invented in advance or reduced to one aggregate F1 score.

## 18. First implementation milestones

1. Pin one EduGraph dataset release and ontology version; generate the dataset profile and leakage audit.
2. Implement the canonical output schema, serializer, validator, and exact-set evaluator with unit tests.
3. Version the direct-labeling prompt and render Fireworks train/evaluation JSONL locally.
4. Add a dry-run Fireworks capability and cost/config preflight.
5. Evaluate eligible untuned and historical models.
6. Launch one default LoRA SFT job on the smallest confirmed tunable VLM.
7. Deploy only long enough to run the immutable evaluation suite; then remove or scale down unused dedicated capacity.
8. Produce the comparison/error report and choose the next experiment from the follow-up ladder.

## 19. References

Provider documentation:

- [Fireworks managed SFT and vision dataset format](https://docs.fireworks.ai/fine-tuning/fine-tuning-models)
- [Fireworks training model eligibility](https://docs.fireworks.ai/fine-tuning/models)
- [Fireworks Training API overview and custom-loss boundary](https://docs.fireworks.ai/fine-tuning/training-api/introduction)
- [Fireworks SFT cookbook](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/sft)
- [Fireworks structured outputs](https://docs.fireworks.ai/structured-responses/structured-response-formatting)
- [Fireworks Qwen3-VL 30B-A3B Instruct catalog entry](https://fireworks.ai/models/fireworks/qwen3-vl-30b-a3b-instruct)
- [Fireworks Qwen3-VL 8B Instruct catalog entry](https://fireworks.ai/models/fireworks/qwen3-vl-8b-instruct)

Relevant research for later custom objectives/architectures:

- [Asymmetric Loss for Multi-Label Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf)
- [Distribution-Balanced Loss for Multi-Label Classification](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf)
- [ML-GCN: graph convolution for multi-label recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.html)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://papers.nips.cc/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)
