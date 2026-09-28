# Qwen3.8-27B training API setup

The current executable experiment uses Fireworks serverless training, dataset **v0.30.0-02**, and ontology **v0.30.0**. The one-image diagnostic on 2026-09-28 completed forward/backward, an optimizer step, checkpointing, and image sampling. Its invalid prediction established pipeline operation only. The user authorized the larger learning comparison described here.

## Observed learning smoke and early stop

The run started on 2026-09-28 UTC and completed on 2026-09-29 local time. On the user's request to stop once learning was evident, training stopped at the first durable epoch checkpoint: **160 images, 40 optimizer steps**, instead of the original three-epoch/120-step maximum. A paired evaluation used the same 64 official-validation images and 16 training-diagnostic images as the baseline.

| Validation metric | Base | One epoch |
|---|---:|---:|
| Exact-set match (primary) | 0/64 | 2/64 |
| Micro precision | 43.5% | 62.1% |
| Micro recall | 23.9% | 73.6% |
| Micro F1 | 30.9% | 67.4% |
| Label-macro F1 | 22.9% | 43.4% |
| Invalid outputs | 17/64 | 5/64 |

Per-image F1 improved on 60 images and declined on four. Both question and solution views and all three dimensions improved. A separate diagnostic strips only enclosing Markdown fences: baseline F1 becomes 35.3%, while tuned F1 stays 67.4%. The improvement therefore includes substantial label-selection gains. Training-diagnostic F1 rose from 30.2% to 69.6%, close to the held-out result. These descriptive results are sufficient evidence of learning for the smoke objective; the remaining two epochs were not run.

The checkpoint is **not production-ready**. Exact sets are correct on only two validation images. Tuned output contains 255 extra and 150 missing dimension/label pairs; the five invalid outputs contain invalid identifiers or dimension assignments. Abilities remain the weakest dimension (48.6% F1), and labels unseen in training reach only 26.7% F1. Follow-up quality work should inspect these errors and use a larger untouched evaluation cohort before deciding on more training or deployment.

The run used code commit `3060f9ff44d6fc2e2c4f55dd4fe58276983f3c7b`. Because that running process had no cooperative stop hook, it was terminated immediately after saving epoch 1. A fresh pooled session restored that checkpoint, performed **zero optimizer steps**, saved a sampler checkpoint, and evaluated it. This also verified cross-session state recovery. At most one extra forward/backward batch could have been submitted before termination; it is excluded from the evaluated checkpoint. The original execution snapshot, stop evidence, hashed recovery script, raw predictions, comparison, and final decision remain in `runs/edugraph-20260928-qwen38-27b-learning-v1/`.

The adapter was retained privately as `accounts/christian-bick-91boy/models/edugraph-20260928-qwen38-27b-learning-v1` and returned READY. The evaluation clients and heartbeat holder were closed; the original process was stopped. No inference deployment was created. Observed completed training tokens (543,097) and all 160 sampling prompt-token counts exactly matched local rendering counts. Estimated token cost was **$3.27**, with a possible extra batch allowance below **$0.07**; billing is not reconciled. The detailed local report is `reports/fireworks-qwen38-27b-learning-20260928.md`.

The tracked recipe preserves the original plan and remains a historical record. New recipes may evaluate after each epoch and stop after a configured number of epochs without validation improvement. The run can also receive an operator stop request that finishes the current epoch, saves durable training state, evaluates its sampler checkpoint, and retains the best evaluated checkpoint.

## Safeguards for the next run

Preparation now writes `closed_schema.json` with separate, alphabetized lists of every eligible Area, Scope, and Ability in the pinned ontology. These are allowed identifiers, not labels inferred from the training sample. Label array order is unrestricted in evaluation; sorting is a deterministic training and display convention. A new recipe may opt into schema-constrained sampling with `evaluation.output_mode: "json_schema"`; both base and tuned sampling then receive the same schema. The adapter passes Fireworks' `response_format` to its tokenized image completions call. This path is unit-tested locally but has **not yet been checked against the live multimodal inference endpoint**. Keep the original smoke's unconstrained responses and metrics separate for comparison. The original three-array schema is unchanged and remains useful for offline validation.

With `evaluation.every_epoch: true`, the run saves both optimizer state and an evaluable sampler checkpoint after every epoch. Selection maximizes official-validation explicit exact-set match, then micro F1, breaking ties in favor of the earlier epoch. Set `training.early_stop_patience` to a positive number of consecutive non-improving epochs. The base and every evaluated epoch are counted in the cost estimate and actual token-use report; selection uses the same validation subset repeatedly, so a separate untouched cohort is required for a final quality claim. These controls are locally tested with a fake provider; no new paid run has used them.

For a new recipe, `training.expected_target_modules` declares the expected provider LoRA suffixes. Preparation and launch compare them with every linear module in the pinned Qwen architecture using meta tensors, reject missing targets or any target outside `model.language_model`, and hash the matched paths. The SDK's `train_attn`, `train_mlp`, and `train_unembed` flags must then be explicit. When Fireworks registers the final adapter, its returned `peftDetails.targetModules` must match the recorded expectation. This is an enforceable configuration check; Fireworks restricts downloading the private adapter's tensor files, so it is not an independent tensor-level audit. The historical smoke's retained adapter reported 496 matching language modules and no vision or bridge targets.

## Full-release candidate and final validation

The proposed, **unlaunched** full-release recipe is `experiments/edugraph-20260929-qwen38-27b-full-v1.json`. It keeps dataset v0.30.0-02 and ontology v0.30.0, trains on all 1,654 official-train images, and reserves 64 official-validation images for checkpoint selection. The other 250 official-validation images form a disjoint `final_validation` cohort, sampled only once after checkpoint selection. Both cohorts are balanced between question and solution views because the pinned release is balanced. This final cohort was not used for optimizer steps or epoch selection, but it is a remainder of the same official validation split, not an independently published test set; no semantic grouping key is exposed upstream.

Preparation downloads and validates all 1,968 release images, checks byte overlap between train and validation and between the two validation cohorts, records duplicate-byte counts, and hashes source images and rendered examples. This is a classifier-side input check, not a replacement for upstream release QA. The candidate enables language-only LoRA target checks, greedy closed-vocabulary JSON schema, per-epoch evaluation, and patience-one early stopping with a three-epoch ceiling. It retains the best checkpoint by explicit exact-set match and then F1. The schema transport still requires a live multimodal compatibility check before paid full training. Provider token prices in the recipe are a dated estimate, not a billing cap; recheck them and the live model catalog before launch.

The local full-release preparation completed against code `bcf68dc` on 2026-09-29: all 1,968 images validated, 1,242 maximum optimizer steps, and a **$74.81** three-epoch upper token estimate using the recipe's pricing snapshot. The target audit found 496 matching language linear modules and zero vision/bridge modules. It also found **one pair of byte-identical training images with different gold label sets**. One row asserts `Addition`, the other `Subtraction` for the same pixels; all other labels agree. The pair and image digest are recorded in gitignored `reports/qwen38-full-preflight-duplicate-conflict.json`. No gold label was changed. This candidate is **blocked for paid launch** pending upstream correction or an explicit documented exclusion policy; new launch code refuses conflicting identical training images before starting a provider session. A new clean-commit preparation will be required after resolution. The official validation partition remains a selection/assessment partition, not an independent test release.

## Reproducibility and cohort

The tracked recipe is `experiments/edugraph-20260928-qwen38-27b-learning-v1.json`.

| Component | Pin |
|---|---|
| Dataset | `christian-bick/edugraph-exercises`, v0.30.0-02, `c1a474c2ed2c8b8b959d3205483ec32a3108c52d` |
| Ontology | Official `edugraph-py` v0.30.0 wheel in `uv.lock` |
| Ontology snapshot | `5a565f7c5c10c2ec7d2704294bdb27183ae34a8aa02112e1dc2278f560e1ec79` |
| Base model | `accounts/fireworks/models/qwen3p8-27b` |
| Tokenizer/processor | `Qwen/Qwen3.8-27B`, `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Prompt/schema | `direct-label` v2; `dimension-labels` v1; resolved contents hashed |
| Execution | Provider `fireworks`, mode `serverless`, training API V2 shared pool |

The manifest records code commit, lock hash, core package versions, full recipe, live capabilities, all source identities, image bytes, processor files, resolved prompt/schema, and rendering fingerprints. No base-model weights are downloaded locally.

The release has 1,654 train and 314 validation rows. Metadata uses `labels`, despite the stale `tags` description in the dataset card. Preparation validates all metadata against the pinned ontology and image integrity for the selected cohort; it is not a full-release image audit. Historical recipes retain their original pins and require their original environment. Ontology v0.30 has semantic changes: old gold is not silently reinterpreted as v0.30 gold.

Select 80 question and 80 solution training images, plus 32 question and 32 solution validation images. Within each official split/view, order by SHA256(seed:opaque_filename), then take the requested count. Preserve official splits and check selected image bytes for overlap. Public metadata has no supported task-group identifier, so semantic independence cannot be established independently of the official split policy. Filenames are not parsed for grouping. Within-split duplicates are reported without rewriting gold.

Train three epochs, batch size four, 120 steps, LoRA rank eight/alpha 16, constant Adam learning rate 0.0001, betas 0.9/0.95, epsilon 1e-8, zero weight decay, seed 42. This is a pilot choice, not a tuned optimum. A local 16,384-token context ceiling rejects oversized examples instead of truncating images or targets.

Sample the base and final checkpoint on the same 64 validation images and fixed 16-image training diagnostic. Both use greedy decoding, seed 42, and at most 512 output tokens. Primary outcome: validation explicit-label exact-set match. Also report micro precision/recall/F1, example-mean and label-macro F1, per-dimension, question/solution, training-frequency slices (unseen, 1–4, 5+), invalid outputs, latency, and estimated cost. This small balanced cohort is not a representative-weighted full-validation result or final test set. No significance or production-quality claim follows from a smoke test.

Raw responses and structural canonicalization are separate. No closure, label substitution, fence stripping, or gold correction occurs. Unparseable JSON misses every gold label. Unknown/misdimensioned labels remain false positives. Duplicate labels invalidate strict exact match; deduplicated exact match is separate. Label-macro F1 covers ontology labels with gold or prediction support. Slice metrics restrict the label universe; their exact-match field is not the primary task metric. Empty precision/recall/F1 denominators yield zero.

## Fixed system prompt and deployment template

Use the **system** role; Qwen needs no `admin` role. The user message contains the fixed short instruction and one image. The assistant target is compact dimension JSON. The resolved system prompt includes the complete eligible ontology vocabulary grouped by dimension, derived from the pinned catalog, not selected gold. Definitions are not included yet. The same prompt is used for base evaluation, SFT, and tuned evaluation.

Only assistant target tokens and its terminator receive loss. System, user, image, and empty-thinking-prefill tokens have zero loss. Reasoning is disabled through the native `enable_thinking=False` template option.

Preparation exports `chat_template.jinja` with the fixed system text embedded and verifies that its generation prefix exactly matches the explicit-system training prefix. An identical caller system message is accepted without duplication; conflicting system text is rejected. Inference must still provide the fixed user text and image. Embedding the system prompt removes request boilerplate, not context tokens, and does not bake the instructions into model weights.

Fireworks supports custom `tokenizer_config.json` chat templates **for base models, not LoRA adapters**. LoRA `fireworks.json` supports generation defaults, not a system prompt. The immediate Fireworks serving path therefore uses a private LoRA on a dedicated deployment with the application injecting the fixed system message. Literal provider-side embedding requires a supported custom base/merged-model package or a serving runtime accepting custom templates. The export is locally verified; that deployment path is not provisioned or validated. See [custom models](https://docs.fireworks.ai/models/uploading-custom-models).

## Commands and lifecycle

```powershell
uv sync --group training-api
uv run --group training-api edugraph-classify training-api prepare
uv run --group training-api edugraph-classify training-api launch `
  --manifest runs/edugraph-20260928-qwen38-27b-learning-v1/manifest.json `
  --confirm-run-id edugraph-20260928-qwen38-27b-learning-v1

# For a new run that is currently active:
uv run --group training-api edugraph-classify training-api request-stop `
  --manifest runs/<new-run-id>/manifest.json `
  --confirm-run-id <new-run-id>

# After a completed one-epoch run, prepare a new run ID with additional epochs.
# Verify the continuation locally before its separate, paid launch:
uv run --group training-api edugraph-classify training-api resume-check `
  --manifest runs/<continuation-run-id>/manifest.json `
  --source-execution runs/<first-run-id>/execution.json `
  --source-epoch 1 --confirm-run-id <continuation-run-id>
uv run --group training-api edugraph-classify training-api resume `
  --manifest runs/<continuation-run-id>/manifest.json `
  --source-execution runs/<first-run-id>/execution.json `
  --source-epoch 1 --confirm-run-id <continuation-run-id>
```

Preparation requires a clean checkout, checks live eligibility read-only, and downloads public data and processor files. Launch requires the same clean commit, unchanged hashes/rendering, current capabilities, and exact run ID confirmation. The exclusive execution journal blocks paid replay, including after ambiguous failure. Failed runs need inspection and deliberate recovery; there is no automatic retraining or dedicated fallback.

V2 eligibility requires READY state, sufficient context, and explicit `supportsImageInput`, `supportsLora`, `useTrainingV2`, and `supervisedLoraTunable`. Fireworks documents `tunable` as a deprecated V1 flag. Legacy preflight still requires it when V2 is not explicit. This updates the earlier `Tunable: true` shorthand; actual vision-training success supplies additional surface evidence. Serverless inference support is separate. See [GetModel](https://docs.fireworks.ai/api-reference/get-model).

Every epoch saves training state (adapter and optimizer). New recipes may also save/evaluate a sampler checkpoint each epoch; the highest-scoring checkpoint is registered as a private experimental model before session teardown. `request-stop` creates a local, run-ID-matched stop file and requires no provider credentials or clean checkout. The running process reads it at the next epoch checkpoint, then completes evaluation and retention; it does not interrupt an in-flight optimizer step. If the process fails before reading it, inspect the journal and provider session manually. Fireworks calls registration “promotion”; here it means artifact retention, not quality acceptance. It does not publish a model or create a deployment. Samplers, service, and heartbeat holder are closed in cleanup; provider IDs and checkpoints are journaled.

Continuation uses a **new prepared run ID** and treats its `training.epochs` as additional epochs. The source must be a completed run with a fully qualified `save_state` training checkpoint for `--source-epoch`; the promoted sampler-only LoRA cannot restore optimizer state. `resume-check` requires no provider credentials and compares pinned dataset/model/ontology, prompt/schema/processor bytes, language-only target layout, optimizer configuration, and every train and checkpoint-selection example. It permits a different `selection.final_validation` count: omit that cohort from the initial one-epoch recipe to leave it untouched until the continuation. The paid `resume` command repeats these checks, starts a fresh serverless session through `create_training_client_from_state_with_optimizer`, evaluates the restored weights as its baseline, and continues at the next epoch's deterministic shuffle. It retains the restored baseline if the added epoch scores worse. The provider-side cross-session restore was observed in the earlier smoke; this new end-to-end CLI path has been tested locally with a fake provider, not yet in a paid resumed run. Checkpoint availability and compatibility are confirmed by Fireworks only when the paid resume call loads it. [Fireworks serverless checkpoint guide](https://docs.fireworks.ai/fine-tuning/training-api/serverless#saving-and-loading-checkpoints).

The 2026-09-28 pricing snapshot is $4.103/M train, $1.86/M uncached prefill, $5.595/M generated tokens. Preparation counts expanded image tokens and rejects a no-cache token estimate over the recipe's configured limit. This is an estimate guard, not a provider billing cap; actual cache discounts and billing are not reconciled. Recheck [pricing](https://fireworks.ai/pricing) and [serverless documentation](https://docs.fireworks.ai/fine-tuning/training-api/serverless) before future runs. Trained private LoRA serving requires a dedicated deployment.

## Implementation and checks

`learning_data` prepares cohorts/artifacts; `qwen_rendering` renders locally without provider SDK types; `evaluation` computes offline metrics; `learning_resume` validates cross-run provenance; `learning_run` orchestrates a replaceable adapter; only `providers/fireworks_training_api` creates SDK chunks and performs provider operations. Windows uses CPU PyTorch/torchvision for image preprocessing; Linux training containers retain CUDA 12.6.

Load the released `tokenizer.json` with `TokenizersBackend`: AutoTokenizer can rebuild a different Qwen pre-tokenizer. Native chat-template rendering enforces training/generation parity. Image spans are expanded before shifting targets. Fireworks' high-level sampler converts inputs to text-only tokens, so the adapter uses its image-capable completions transport, with one placeholder and corresponding JPEG. This avoids the earlier diagnostic's cookbook monkeypatches and Unix-only imports.

```powershell
uv run --group training-api pytest --cov=edugraph_classify --cov-report=term-missing
uv build
```

Unit tests fake all provider calls and exercise the complete lifecycle, loss masking, system-template parity, evaluation validity, hash/split/cost guards, cleanup, and replay protection. A local check with the real pinned tokenizer/processor additionally verifies image expansion and template parity. Generated run records and detailed reports are under gitignored `runs/` and `reports/`.
