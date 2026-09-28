# Qwen3.8-27B training API setup

The current executable experiment uses Fireworks serverless training, dataset **v0.30.0-02**, and ontology **v0.30.0**. The one-image diagnostic on 2026-09-28 completed forward/backward, an optimizer step, checkpointing, and image sampling. Its invalid prediction established pipeline operation only. The user authorized the larger learning comparison described here.

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
```

Preparation requires a clean checkout, checks live eligibility read-only, and downloads public data and processor files. Launch requires the same clean commit, unchanged hashes/rendering, current capabilities, and exact run ID confirmation. The exclusive execution journal blocks paid replay, including after ambiguous failure. Failed runs need inspection and deliberate recovery; there is no automatic retraining or dedicated fallback.

V2 eligibility requires READY state, sufficient context, and explicit `supportsImageInput`, `supportsLora`, `useTrainingV2`, and `supervisedLoraTunable`. Fireworks documents `tunable` as a deprecated V1 flag. Legacy preflight still requires it when V2 is not explicit. This updates the earlier `Tunable: true` shorthand; actual vision-training success supplies additional surface evidence. Serverless inference support is separate. See [GetModel](https://docs.fireworks.ai/api-reference/get-model).

Every epoch saves training state (adapter and optimizer). The final sampler checkpoint is separately saved and registered as a private experimental model before session teardown. Fireworks calls this registration “promotion”; here it means artifact retention, not quality acceptance. It does not publish a model or create a deployment. Samplers, service, and heartbeat holder are closed in cleanup; provider IDs and checkpoints are journaled. Fireworks supports optimizer-state resume, but an automated resume CLI is not implemented.

The 2026-09-28 pricing snapshot is $4.103/M train, $1.86/M uncached prefill, $5.595/M generated tokens. Preparation counts expanded image tokens and rejects a no-cache token estimate over the recipe's $20 limit. This is an estimate guard, not a provider billing cap; actual cache discounts and billing are not reconciled. Recheck [pricing](https://fireworks.ai/pricing) and [serverless documentation](https://docs.fireworks.ai/fine-tuning/training-api/serverless) before future runs. Trained private LoRA serving requires a dedicated deployment.

## Implementation and checks

`learning_data` prepares cohorts/artifacts; `qwen_rendering` renders locally without provider SDK types; `evaluation` computes offline metrics; `learning_run` orchestrates a replaceable adapter; only `providers/fireworks_training_api` creates SDK chunks and performs provider operations. Windows uses CPU PyTorch/torchvision for image preprocessing; Linux training containers retain CUDA 12.6.

Load the released `tokenizer.json` with `TokenizersBackend`: AutoTokenizer can rebuild a different Qwen pre-tokenizer. Native chat-template rendering enforces training/generation parity. Image spans are expanded before shifting targets. Fireworks' high-level sampler converts inputs to text-only tokens, so the adapter uses its image-capable completions transport, with one placeholder and corresponding JPEG. This avoids the earlier diagnostic's cookbook monkeypatches and Unix-only imports.

```powershell
uv run --group training-api pytest --cov=edugraph_classify --cov-report=term-missing
uv build
```

Unit tests fake all provider calls and exercise the complete lifecycle, loss masking, system-template parity, evaluation validity, hash/split/cost guards, cleanup, and replay protection. A local check with the real pinned tokenizer/processor additionally verifies image expansion and template parity. Generated run records and detailed reports are under gitignored `runs/` and `reports/`.
