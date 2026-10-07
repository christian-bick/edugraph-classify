# EduGraph classifier: design and experiment direction

Status (2026-10-03): full baseline established; targeted refinement research planned. The [refinement plan](REFINEMENT-PLAN.md) records the current evidence, priorities and evaluation boundaries. Earlier dated decisions below retain their original pins and are historical where superseded. Historical recipe and prompt paths identify files removed from the current checkout; the original commits preserve them for reproduction.

Accepted dataset priorities (2026-10-03): one consistent realized-task range policy and complete coverage of legitimate, observable labels across task generators and views. The [upstream handoff](UPSTREAM-ANNOTATION-HANDOFF-20261003.md) contains source-level analyses and implementation acceptance checks. This supersedes treating omission of an independently supported facet as an acceptable distinguishability convention; it does not mutate historical releases or settle every individual semantic case.

Current reference: **Qwen3.8-27B NF4 QLoRA on Runpod Secure A40**, dataset **v0.30.0-03**, ontology **0.30.0**, compact v3 system prompt, W&B tracking and GCS recovery artifacts. The completed two-epoch run achieved **72.4% exact-set match and 96.44% label F1 on 250 final images**, with zero invalid outputs. Only language adapters trained; vision and bridge stayed frozen. Model and continuation artifacts were independently verified, the Pod terminated automatically, and the candidate remains undeployed. See the [full-run record](RUNPOD-FULL-A40-20261002.md) for provenance and [Runpod training](RUNPOD-TRAINING.md) for execution policy.

Container registry decision (2026-10-02): use **GHCR** for new self-hosted training images, retaining GCS for run artifacts and **GCP scale-to-zero for production inference**. Publication is a manual workflow with CPU import/CLI checks, commit tags, and an immutable digest in the result. The authorized [GCP cleanup](GCP-CLEANUP.md) removed four obsolete training repositories (164.65 GiB); production and rollback image repositories remain. Historical training image references remain pinned for provenance but no longer resolve to the deleted images. See [container images](CONTAINER-IMAGES.md).

Archived Fireworks candidate (2026-10-02): user-selected **Qwen3.8-27B**, dataset **v0.30.0-03**, ontology **v0.30.0**. Its completed 160-image learning smoke and one-image diagnostic remain pinned to v0.30.0-02. The compact prompt, closed schema, generated checkpoint selection and language-only LoRA policy informed the Runpod executor. See the [Training API record](FIREWORKS-TRAINING-API.md) and [training costs](TRAINING-COSTS.md) for comparison evidence. The Runpod final cohort has now been examined for error analysis; the refinement plan governs subsequent reuse.

Initial model hypothesis: Qwen3.5-9B

Initial managed provider: Fireworks AI

First executable baseline: Qwen3-VL-8B-Instruct

Historical local-training plan (2026-09-13): user-selected Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, with QLoRA on both RTX 3090s from the first diagnostic. DDP was the initial backend; two-GPU FSDP-QLoRA the memory fallback. This earlier plan remains in [Local Docker training](LOCAL-TRAINING.md); the current reference is the completed Runpod 27B run above.

## 1. Core decision

Historical data preparation (2026-09-13): the local 9B recipe pinned dataset v0.26.0-01 at `cee47a3b49503e2637759a8a0e59e071c271b415`, retaining its source ontology v0.26.0 while validating with the semantically identical v0.27.0 client. Its [release audit](DATASET-UPGRADE-0.26.0-01.md) remains specific to that release. The current baseline and refinement plan use the full-run record's v0.30.0-03/0.30.0 identities.

Use direct vision-language labeling of one isolated educational task as the reference experiment:

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

Raw predictions, validated explicit predictions, and any ontology-derived expansion remain separate. Pure `specializes` paths support taxonomic inheritance; recorded `implies` relations separately support logical closure. `partOf` provides structural context, and progression relations such as `integrates` justify no automatic label addition or removal. Nodes with constituent children are organizational and are ineligible for direct labeling. Observable specificity remains the goal, but the baseline preserves released explicit gold without ancestor pruning; suspected convention conflicts require upstream review and a versioned policy, not silent target repair.

## 3. Model selection and historical starting hypothesis

The established reference is **Qwen3.8-27B**, with frozen vision/bridge and language-only QLoRA. The original **Qwen3.5-9B** hypothesis and the first executable Qwen3-VL-8B provider experiment are historical decisions, retained below for provenance.

Before a future provider/model comparison, verify the exact checkpoint's required vision inference and training capabilities. Availability is a live preflight result, not an assumption embedded in core code. Model or provider changes are explicit experiment decisions, supported by a concrete research question rather than silent substitution.

The kickoff does not prescribe LoRA rank, epochs, learning rate, context size, prompt wording, JSON representation, or deployment lifetime. Those choices can be made during implementation from the actual model/provider surface and recorded in the resulting experiment manifest.

## 4. Provider adapters and execution modes

Provider-neutral classifier logic is a durable requirement. Provider-specific SDK objects and lifecycle concepts belong behind thin adapters.

```text
dataset + ontology + evaluation core
                 |
                 v
           adapter boundary
            /            \
 Runpod adapter       future adapter
       |                   |
self-hosted         explicit mode
```

The neutral core owns dataset ingestion, ontology validation, prediction contracts, canonicalization, metrics, and reproducibility metadata. A provider adapter translates those contracts into model discovery, training, inference, artifact, and deployment operations for a particular provider.

Provider choice may differ by operation. Current training uses Runpod Secure with self-hosted execution; the historical local Docker path is another self-hosted executor. New container images use GHCR, while input staging and model/artifact uploads remain on GCS. Record these responsibilities separately; GCS use does not make a training run a Vertex job. Shared artifact operations remain reusable across executors. Earlier GCP container and downstream-service decisions below describe their historical scope.

Execution mode is recorded separately from provider identity:

- **Serverless:** inference is invoked through a provider-managed shared service with provider-defined capabilities and lifecycle.
- **Provider-hosted dedicated:** a provider manages a training job or dedicated deployment. This is managed infrastructure, but it is not serverless and should not be mislabeled as self-hosted.
- **Self-hosted:** EduGraph controls the model runtime and compute environment, whether local or on rented cloud infrastructure.

The Runpod executor uses provider-neutral preparation, output contracts and evaluation. Refinement experiments must preserve that boundary so an inference or training adapter can change without redefining the data or metrics. The Fireworks adapter described in older sections is archived.

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

- Can the selected model learn the complete explicit label set, including infrequent concepts and unfamiliar combinations?
- Where is the balance between missing labels and plausible but unsupported extra labels?
- How does performance differ between unsolved questions and worked solutions, accounting for their different content?
- Which failures are related to rare labels, cross-dimension combinations, mathematical inference, output validity, or ontology canonicalization?
- What quality, latency, and cost does the selected execution path provide?

Exact-set match is the primary metric. Precision, recall, F1, per-dimension results, question/solution slices, rare-label coverage, invalid outputs, latency, and cost provide the diagnosis. Raw and canonicalized predictions are both retained so deterministic post-processing gains remain visible.

The initial evaluator should compare an untuned checkpoint and its fine-tuned counterpart under equivalent conditions. Other model comparisons are useful only when they answer a concrete question.

The first full run now provides that comparison. Its error analysis identifies numerical specificity, required cognitive action and fraction-related distinctions as priorities. All final gold combinations occur in training, and the released metadata has no supported task-group identity; unseen-combination generalization and semantic independence remain unestablished. See the [evidence and limits](REFINEMENT-PLAN.md#1-evidence-and-its-limits).

## 7. Reasoning

Reasoning may matter most when an unsolved task does not visually state the relevant operation or procedure. A model may need to solve or mentally simulate the task before selecting the correct labels. Worked solutions may expose the same evidence directly.

This can first be investigated at inference time without reasoning fine-tuning, while scoring only the final label set. If later evidence supports reasoning supervision, the useful target is concise and verifiable evidence connected to the retained labels, with the final canonical labels kept separate. Free-form reasoning is not assumed to be beneficial: it may also introduce related but unsupported concepts and reduce precision.

The current baseline does not establish a general question-versus-solution difficulty ordering. Prefer targeted tests that identify visible inputs/results, the unknown quantity or the requested cognitive action, after clarifying ambiguous label conventions. Measure final-label quality and added latency against direct prediction; do not turn an unverified explanation into new gold supervision.

## 8. Current refinement priorities

The [refinement plan](REFINEMENT-PLAN.md) orders the next studies from the observed error distribution:

1. Audit numerical bounds and other evidence/serialization ambiguities with the upstream owners.
2. Create controlled contrast examples for the resolved scope, operation/strategy and ability distinctions.
3. Establish new development/challenge data and a fresh final assessment with supported grouping, unfamiliar combinations and visual styles.
4. Test narrowly retrieved definitions or short verifiable task evidence while retaining the compact direct baseline.
5. Compare training duration, learning rate and targeted sampling on development data, recording regressions as well as gains.

Checkpoint disagreement is a candidate human-review signal to validate independently. The existing final cohort has now informed these hypotheses; future methods selected from it need fresh confirmation. Keep exact-set match primary and report label support, coverage and positive controls. Do not restrict predictions to label combinations observed in training.

Additional format repair, blanket reasoning, hard graph correction, larger models and reward/preference training are lower priorities under current evidence. Keep vision and bridge frozen. Discriminative heads and graph-aware decoders remain options for latency, calibrated scores or a demonstrated accuracy limitation, with calibration, imbalance and ontology-migration work included in the comparison. Candidate filtering must preserve recall; ontology neighborhoods and empirical correlations cannot become unsupported hard constraints. “The direct model is sufficient” remains a valid result.

## 9. Durable engineering principles

- Keep provider calls behind adapters and keep offline data/evaluation logic usable without credentials.
- Record provider, execution mode, exact model identity, dataset release, ontology version, prompt/schema identity, code commit, and resolved run configuration.
- Preserve raw predictions alongside validated and derived representations.
- Make conversion, canonical serialization, metrics, and report inputs deterministic.
- Keep credentials, provider-ready datasets, model artifacts, and generated reports out of Git.
- Make paid training and deployments explicit operations that require user confirmation.

## 10. Immediate research scope

The baseline implementation and first full run are complete. The immediate research work is to document the upstream evidence questions, define controlled contrasts and fresh assessment requirements, and specify bounded development experiments. The [refinement plan](REFINEMENT-PLAN.md#3-ordered-research-plan) records their order and evidence requirements. Changes to data, model or training policy must be separately versioned; paid runs, uploads and deployments retain their explicit authorization boundary.

## 11. Historical accepted implementation defaults

The following decisions describe the initial implementation. Dated dataset and model pins, provider adapters, recipes and CLI commands in this section are historical; the current baseline identity and supported Runpod workflow are recorded above. The durable provider-neutral contracts remain relevant.

### Data and ontology pinning

Use the latest public `christian-bick/edugraph-exercises` release on Hugging Face as the baseline source. Resolve the moving release reference to its full immutable Hub commit before conversion and record that commit in every run identity. Use the exact ontology release declared by the dataset; fail closed if its ontology provenance cannot be resolved rather than validating against an unpinned latest ontology.

The initial baseline pair is dataset tag `v0.21.0-01` at commit `ed47264f751a8a67c480dbe4eb7397e317a722eb` with ontology release `v0.21.0`. Treat this pair as run configuration and consume it directly through the dataset library; Hugging Face is a data source, not a classifier provider adapter.

As of 2026-09-13 the development package pins ontology client v0.27.0. Its authored ontology and normalized classifier snapshot are unchanged from v0.26.0, so the aligned v0.26.0-01 dataset remains valid while retaining its source version in the recipe. The initial experiment files and their v0.21.0 pins are retained only at the original code commits; those historical recipes would fail the current ontology version check. Neither the initial dataset nor the newer v0.22.2 dataset is approved as v0.26.0 gold data; see [the v0.26.0 migration assessment](ONTOLOGY-UPGRADE-0.26.md) and [v0.27.0 adoption notes](ONTOLOGY-UPGRADE-0.27.md). Historical reproduction requires the corresponding code commit and lockfile. The [local Docker training assessment](LOCAL-TRAINING.md) proposes a `local_docker` / `self_hosted` training adapter while retaining GCP for all surrounding services, including model upload after training. The assessment does not launch a run.

Public dataset discovery and download do not require a Hugging Face credential. Authentication is introduced only if a future source is private or gated, or when an explicit model-publication step is approved.

### Model capability preflight

For a new Fireworks experiment, resolve the selected model's exact identity through a live, read-only capability check. Eligibility requires vision inference plus a compatible fine-tuning surface reporting `Tunable: true`. If no eligible checkpoint exists, stop and report the observed capabilities and alternatives; do not silently substitute a nearby model.

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

## 12. First executable baseline decision (2026-08-23)

The original Qwen3.5-9B hypothesis failed the required live Fireworks preflight because the catalog did not report `Tunable: true`. The subsequently requested Qwen3 4B checkpoint was tunable but text-only, so it could not consume the released task images. Neither checkpoint was silently substituted.

With explicit approval, the first executable baseline is `accounts/fireworks/models/qwen3-vl-8b-instruct`. The live catalog check confirmed vision input, `Tunable: true`, LoRA support, and supervised LoRA eligibility. It did not report serverless inference support, so any later inference deployment is a distinct provider-dedicated and cost-incurring decision outside this run.

The tracked experiment is `experiments/edugraph-20260823-qwen3vl8b-sft-v1.json`. It uses all 1,602 official training examples and all 356 official validation examples from the pinned release. Each assistant target is canonical JSON with separate `areas`, `scopes`, and `abilities` arrays; images are embedded in the provider's multimodal chat format after integrity, type, size, path, split, ontology, and duplicate checks.

Exact image duplicates are treated according to their effect on the experiment contract. A duplicate crossing the official train/validation boundary fails preparation because it would leak evaluation input. Records duplicated within one official split are preserved rather than changing upstream gold data; the profile records every duplicate group and whether its gold label sets conflict. Such conflicts are upstream-data diagnostics and must be considered when interpreting achievable exact-set accuracy.

The first training shape is deliberately conservative:

- supervised LoRA, one epoch, rank 8;
- provider-selected learning rate and maximum context length, with the resolved values captured from the created job;
- the official validation split supplied explicitly, with early stopping and automatic validation carveout disabled;
- unique Fireworks dataset, job, and output-model identifiers so existing resources are never overwritten.

Preparation and launch are separate operations. Preparation is offline with respect to Fireworks, needs no Hugging Face credential for this public dataset, records a clean code commit plus all source and output hashes, and stores generated files only under gitignored `runs/`. Launch reloads `FIREWORKS_API_KEY` at the application boundary without logging it, repeats the exact-model capability check, verifies artifact hashes and resource collisions, then uploads the two datasets and starts one job. Launch does not deploy or publish the trained model.

### Technical smoke run

The pinned full release currently fails the classifier's required cross-split byte-identity safeguard. Until a corrected upstream release is available, `experiments/edugraph-20260823-qwen3vl8b-smoke-v1.json` is authorized only to validate the managed-training plumbing. It deterministically selects the first eight source paths from each official split, while retaining all ordinary row, image, ontology, and selected-subset cross-split checks. It does not repair labels, move examples between splits, or establish a quality result.

The smoke job uses the same Qwen3-VL-8B-Instruct, prompt, schema, one-epoch LoRA rank 8 shape, and no-deployment boundary as the intended baseline. Its metrics and resulting model must not be compared or promoted as the baseline; success means only that conversion, provider dataset upload and validation, and supervised fine-tuning job creation work end to end.

Provider mutations are checkpointed before each upload and job-creation call. A retry may resume only resource names recorded by the same manifest, verifies an existing dataset's identity and expected example count, and otherwise preserves the no-overwrite rule. The first `smoke-v1` launch encountered a connection failure after creating its training dataset. The retry-safe `smoke-v2` run reproduced the failure and isolated it to an SDK multipart tuple; a direct `Path` upload succeeded. Both partial `UPLOADING` resources are retained for audit, the adapter now uses the SDK's native path transport, and the final validation run uses fresh `smoke-v3` identifiers.

`smoke-v3` verified deterministic conversion, both Fireworks uploads, automatic provider validation, and resume checkpoints: its train and validation datasets each reached `READY` with eight examples. Fireworks then rejected managed SFT creation with code `3`, `model is not supported for fine-tuning`. No job, output model, or deployment was created. This response is authoritative over the contradictory Qwen3-VL catalog flags, so Qwen3-VL-8B-Instruct is not an eligible managed-SFT baseline on the observed provider surface. Fireworks' official managed-VLM documentation currently names the Qwen2.5-VL 3B, 7B, 32B, and 72B Instruct family; changing to one of those is a new explicit experiment decision.

### Vertex AI serverless custom-training adapter

After the Fireworks rejection, Google Vertex AI CustomJob is the second managed-training provider path. It is recorded as provider `gcp_vertex_ai` with execution mode `serverless`: Google owns provisioning and teardown, while the run still declares the short-lived machine, accelerator, disk, replica, and scheduling shape. It is not a persistent VM or a self-hosted runtime.

The adapter targets the regional JobService API and defaults by policy—not silently in code—to `europe-west4` for the first EU-resident setup. Storage, Artifact Registry, CustomJob, and output resources should remain in the same region. Google is not an EU-owned provider; this decision concerns workload and artifact location only.

Every resolved Vertex job configuration must pin the clean code commit and immutable training container digest, identify the runtime service account and regional GCS output prefix, and provide bounded compute and scheduling values. Configurations may contain only non-secret container environment variables. Local submission uses Application Default Credentials loaded at the application boundary; API keys and credential contents are never placed in the job configuration or launch record.

Offline rendering and paid submission are separate operations. Rendering validates the configuration and produces the exact `create_custom_job` request without importing credentials or calling GCP. Submission requires an exact job-ID confirmation, rechecks the clean commit, searches the region for the adapter's protected job label, checkpoints the pending mutation, and resumes an unambiguous created job instead of duplicating it after a connection failure. No Vertex job was submitted while establishing this adapter.

Vertex CustomJob executes a supplied container rather than providing a model-specific managed tuning surface. The provider adapter therefore does not choose Qwen, Gemma, or DeepSeek training code. A subsequent experiment decision must select and pin the first eligible 2026 model revision, implement the provider-neutral trainer image, stage the resolved run inputs, and choose a GPU shape supported in the selected region before a smoke launch is authorized.

The initial GCP infrastructure identity is project `edugraph-438718` with empty bucket `gs://edugraph-classify` in `europe-west4`. Vertex AI, Artifact Registry, and Cloud Storage APIs are enabled; the bucket is regional with uniform bucket-level access and public-access prevention; and host Application Default Credentials have the required CustomJob, quota-consumer, Artifact Registry upload, and staging-object permissions. With explicit approval, bootstrap created the keyless runtime service account `vertex-training@edugraph-438718.iam.gserviceaccount.com` and standard Docker repository `europe-west4-docker.pkg.dev/edugraph-438718/training`. IAM grants the runtime identity `roles/storage.objectAdmin` only on the staging bucket and grants the submitting user `roles/iam.serviceAccountUser` only on that service account; effective `iam.serviceAccounts.actAs` was verified. No service-account key, image, dataset, model, or training job was created.

### First Vertex model and training decision

The first Vertex technical smoke run uses the public, ungated `Qwen/Qwen3.5-4B` repository at immutable revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, created on 2026-02-27. The checkpoint is a 4B causal language model with a vision encoder, is tagged for image-to-text use, and operates in thinking mode by default. It therefore satisfies the 2026, 3B–12B, VLM, and reasoning-capable target without substituting the earlier text-only Qwen3 4B model. Because both the model and dataset are public and ungated, this workflow needs no Hugging Face credential; a token is introduced only for a future private/gated dependency or an explicitly approved publication step.

The tracked configuration is `experiments/edugraph-20260823-qwen35-4b-vertex-smoke-v1.json`. It selects the same deterministic eight training and eight validation examples used for plumbing checks and is not a model-quality result. The recipe uses one epoch of 4-bit NF4 QLoRA with bfloat16 compute, gradient checkpointing, rank 8, alpha 16, dropout 0.05, learning rate `2e-5`, per-device batch size 1, and four-step gradient accumulation. LoRA targets all linear layers so the recipe does not hard-code private architecture module names. Prompt and image tokens are masked from loss; only the canonical dimension-aware assistant JSON is supervised.

Qwen3.5's thinking capability is retained in the pinned base model, but the training chat template explicitly disables thinking. The dataset contains verified final labels rather than reasoning traces, and manufacturing hidden chain-of-thought would violate the data contract. Thinking and non-thinking inference can be compared later under the same evaluation contract without changing gold targets.

The first-fit compute hypothesis is one `g2-standard-12` worker with one 24 GB NVIDIA L4, a 200 GB SSD boot disk, and Flex Start scheduling bounded to two hours of wait plus two hours of execution. QLoRA, batch size 1, and gradient checkpointing are selected to fit this shape; it remains an empirical technical hypothesis, not a guarantee. An out-of-memory result changes the recorded compute/recipe decision rather than silently truncating image or target tokens.

Prepared JSONL and runtime metadata are staged under a commit- and hash-addressed GCS prefix. Uploads use create-only generation preconditions and accept a retry only when object size and SHA-256 metadata match. The runtime manifest pins the dataset, ontology, prompt, schema, code, model, recipe, inputs, and output prefix. The trainer preserves GCP and Hugging Face details at its storage/model-loading boundaries; the learning loop consumes the same provider-neutral chat contract that local validation uses.

The container uses Python 3.12.13 plus locked `torch==2.13.0`, `torchvision==0.28.0`, `transformers==5.15.1`, `peft==0.20.0`, `accelerate==1.14.0`, and `bitsandbytes==0.50.1`. The final Vertex configuration can be generated only after the image is published and resolved to an Artifact Registry `@sha256` digest. Staging, image publication, and CustomJob submission remain separate mutations; the job is not submitted without a final run-specific cost estimate and exact job-ID confirmation.
