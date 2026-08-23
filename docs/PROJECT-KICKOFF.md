# EduGraph classifier: kickoff brief

Status: accepted starting direction

Initial model hypothesis: Qwen3.5-9B

Initial managed provider: Fireworks AI

## 1. Core decision

Start with the smallest credible experiment: fine-tune a vision-language model to assign the explicit EduGraph ontology labels supported by one isolated educational task.

```text
atomic task image
        |
        v
 direct VLM labeling
        |
        v
validated explicit label set
```

The baseline exists to establish an honest reference for label quality, cost, latency, and operational complexity. It is not a commitment to generative classification as the final architecture. More specialized reasoning, ontology-aware, discriminative, or self-hosted approaches remain open and should be evaluated against evidence from the baseline.

## 2. Stable problem contract

### One task is one classification unit

The base assumption is:

> One independent pedagogical task equals one classification unit.

Worksheets, documents, and videos containing several independent tasks are segmented before classification. The classifier should spend its capacity identifying the task's mathematical evidence, not discovering task boundaries.

### This is multilabel classification

Each image can require several labels across Area, Scope, and Ability. The quality terms are:

- **Precision:** predicted labels are supported by the task; extra labels reduce precision.
- **Recall:** all gold labels are returned; missing labels reduce recall.
- **Exact-set match:** the predicted explicit set equals the gold explicit set. This is the primary outcome because it requires precision and recall simultaneously.

Area, Scope, and Ability are strongly correlated. The model should learn their joint compatibility, while evaluation should also expose each dimension separately. Correlation is evidence, not a hard rule: a rare combination must not become impossible merely because it was uncommon in one release.

### Predict explicit evidence, derive ontology knowledge separately

The classifier predicts the most specific concepts independently demonstrated by the task. It should not repeat broader facts that can be recovered mechanically from the pinned ontology.

Raw predictions, validated explicit predictions, and any ontology-derived expansion remain separate. In particular, `partOf` can support ancestor handling, while `integrates` describes composition and must not be treated as automatic logical entailment.

## 3. Starting model hypothesis

The first model hypothesis is **Qwen3.5-9B**, not a Qwen3 model. Earlier Qwen3 or Qwen3-VL experiments may be useful historical comparisons, but they are not the intended starting baseline.

Fireworks AI is the first managed provider to test. Before work depends on this combination, a live capability check must confirm that the exact Qwen3.5-9B checkpoint supports the required vision inference and fine-tuning surfaces. Availability is a preflight result, not an assumption embedded in core code. If the combination is unavailable or materially unsuitable, changing it should be an explicit experiment decision rather than a silent substitution.

The kickoff does not prescribe LoRA rank, epochs, learning rate, context size, prompt wording, JSON representation, or deployment lifetime. Those choices can be made during implementation from the actual model/provider surface and recorded in the resulting experiment manifest.

## 4. Provider adapters and execution modes

Provider-neutral classifier logic is a durable requirement. Provider-specific SDK objects and lifecycle concepts belong behind thin adapters.

```text
dataset + ontology + evaluation core
                 |
                 v
           adapter boundary
            /            \
 Fireworks adapter    self-hosted adapter
    /       \                 |
serverless  dedicated     owned runtime
```

The neutral core owns dataset ingestion, ontology validation, prediction contracts, canonicalization, metrics, and reproducibility metadata. A provider adapter translates those contracts into model discovery, training, inference, artifact, and deployment operations for a particular provider.

Execution mode is recorded separately from provider identity:

- **Serverless:** inference is invoked through a provider-managed shared service with provider-defined capabilities and lifecycle.
- **Provider-hosted dedicated:** a provider manages a training job or dedicated deployment. This is managed infrastructure, but it is not serverless and should not be mislabeled as self-hosted.
- **Self-hosted:** EduGraph controls the model runtime and compute environment, whether local or on rented cloud infrastructure.

The first implementation only needs the Fireworks path, but its adapter boundary must not leak into provider-neutral modules. No speculative self-hosted stack needs to be built now; the code should simply preserve the distinction so a later adapter or runtime does not require rewriting data and evaluation logic.

## 5. Repository boundaries

### `edugraph-dataset`

Owns released labeled images, official splits, public metadata, and the evidence contract for each image's complete label conjunction.

### `edugraph-ontology`

Owns label identifiers, dimensions, definitions, versions, and semantic relations.

### `edugraph-classify`

Owns classifier prompts, dataset adaptation, ontology-aware validation, provider adapters, training and inference orchestration, evaluation, and experiment records.

This repository must not redefine upstream gold labels or ontology semantics. Suspected upstream defects are tracked separately from model errors. Dataset releases and ontology versions are pinned together for every reproducible run.

## 6. What the direct baseline must reveal

The baseline should answer a small set of questions:

- Can Qwen3.5-9B learn the complete explicit label set rather than only frequent or visually obvious concepts?
- Where is the balance between missing labels and plausible but unsupported extra labels?
- How much harder are unsolved questions than worked solutions?
- Which failures are related to rare labels, cross-dimension combinations, mathematical inference, output validity, or ontology canonicalization?
- What quality, latency, and cost does the managed execution path provide?

Exact-set match is the primary metric. Precision, recall, F1, per-dimension results, question/solution slices, rare-label coverage, invalid outputs, latency, and cost provide the diagnosis. Raw and canonicalized predictions are both retained so deterministic post-processing gains remain visible.

The initial evaluator should compare an untuned checkpoint and its fine-tuned counterpart under equivalent conditions. Other model comparisons are useful only when they answer a concrete question.

## 7. Reasoning

Reasoning may matter most when an unsolved task does not visually state the relevant operation or procedure. A model may need to solve or mentally simulate the task before selecting the correct labels. Worked solutions may expose the same evidence directly.

This can first be investigated at inference time without reasoning fine-tuning, while scoring only the final label set. If later evidence supports reasoning supervision, the useful target is concise and verifiable evidence connected to the retained labels, with the final canonical labels kept separate. Free-form reasoning is not assumed to be beneficial: it may also introduce related but unsupported concepts and reduce precision.

## 8. Outlook beyond direct VLM classification

Direct labeling is the reference point, not the limit of the project. Plausible directions include:

- improving data coverage, sampling, prompts, or model scale;
- using inference-time or supervised reasoning for tasks that require mathematical derivation;
- retrieving ontology definitions or nearby concepts when the full vocabulary becomes difficult to distinguish;
- using ontology structure for validation, candidate generation, reranking, or joint selection while preserving recall;
- aligning training more directly with set quality through weighting, preference learning, or reward-based methods;
- replacing generative labels with discriminative multilabel heads, label embeddings, or graph-aware decoders when calibrated scores or fixed-vocabulary efficiency become more important;
- moving selected workloads from provider-managed execution to self-hosted training or serving when control, cost, or unsupported model changes justify it.

None of these is selected in advance. Candidate filtering, for example, can reduce work but can also impose a hard recall ceiling. A custom multilabel head can provide direct scores but introduces calibration, class-imbalance, and ontology-migration questions. Ontology-aware decoding can improve consistency but must not turn statistical correlations or `integrates` relations into false logical rules.

The next architecture should follow the observed error distribution and operational constraints of the baseline. “The direct model is sufficient” remains a valid result.

## 9. Durable engineering principles

- Keep provider calls behind adapters and keep offline data/evaluation logic usable without credentials.
- Record provider, execution mode, exact model identity, dataset release, ontology version, prompt/schema identity, code commit, and resolved run configuration.
- Preserve raw predictions alongside validated and derived representations.
- Make conversion, canonical serialization, metrics, and report inputs deterministic.
- Keep credentials, provider-ready datasets, model artifacts, and generated reports out of Git.
- Make paid training and deployments explicit operations that require user confirmation.

## 10. Immediate scope

The first implementation phase should establish only what is needed to run and evaluate the direct baseline:

1. confirm the live Qwen3.5-9B and Fireworks capability combination;
2. define provider-neutral request, prediction, and run identities;
3. connect Fireworks through an adapter;
4. convert and validate a pinned dataset/ontology pair;
5. evaluate the untuned and fine-tuned model with the same set-based metrics.

Implementation details that do not affect these durable boundaries can be decided as they arise and documented at the point they become real decisions.

## 11. Accepted implementation defaults

The following decisions define the initial implementation without changing the durable architecture above.

### Data and ontology pinning

Use the latest public `christian-bick/edugraph-exercises` release on Hugging Face as the baseline source. Resolve the moving release reference to its full immutable Hub commit before conversion and record that commit in every run identity. Use the exact ontology release declared by the dataset; fail closed if its ontology provenance cannot be resolved rather than validating against an unpinned latest ontology.

The initial baseline pair is dataset tag `v0.21.0-01` at commit `ed47264f751a8a67c480dbe4eb7397e317a722eb` with ontology release `v0.21.0`. Treat this pair as run configuration and consume it directly through the dataset library; Hugging Face is a data source, not a classifier provider adapter.

Public dataset discovery and download do not require a Hugging Face credential. Authentication is introduced only if a future source is private or gated, or when an explicit model-publication step is approved.

### Model capability preflight

Resolve the exact Fireworks model identity behind the Qwen3.5-9B hypothesis through a live, read-only capability check. Eligibility requires vision inference plus a compatible fine-tuning surface reporting `Tunable: true`. If no eligible checkpoint exists, stop and report the observed capabilities and alternatives; do not silently substitute a nearby model.

### Prediction contract

Use a dimension-aware response with `areas`, `scopes`, and `abilities` arrays. The arrays have set semantics and canonical serialization uses a fixed property order plus deterministic identifier ordering.

Preserve four distinct prediction views:

1. **Raw:** the provider response and relevant response metadata exactly as received.
2. **Parsed:** the structured explicit labels extracted from the raw response.
3. **Canonical:** the validated, deterministically ordered explicit label set after documented cleanup.
4. **Derived:** separately computed ontology closure or other logically derived facts.

Never overwrite one view with another or attribute deterministic post-processing gains to the model.

### Core and adapter boundary

Define provider-neutral request, prediction, model, execution-mode, and run identities in the core package. Provider SDK types remain inside thin adapters. In particular, provider identity and execution mode are independent fields rather than one combined provider-specific state.

The first adapter interface targets managed training providers. Each adapter reports its provider identity and training execution mode explicitly, so a future serverless training surface and Fireworks provider-hosted training can implement the same neutral capability contract without conflating their infrastructure lifecycles.

### Configuration and secrets

Load local `.env` configuration explicitly at application entry points without overriding environment variables already supplied by the process. Never print, persist, or include secret values in errors, manifests, reports, or fixtures. `FIREWORKS_API_KEY` is used only by the Fireworks boundary.

### Test tooling

Manage pytest and coverage tooling through uv and keep `pyproject.toml` and `uv.lock` synchronized. Production provider calls are tested with fakes or recorded non-secret fixtures, while live checks remain explicit read-only integration operations.
