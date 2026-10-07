# Trained model to Imagine production

This is the working release path for the Qwen3.8-27B classifier and the template for a later trained model. Training and evaluation live here; `imagine-server` owns the application and predictor deployments, and `imagine-web` owns the public browser bundle. A benchmark, Hub publication, image build, tagged canary, and web cutover are separate operations. Keep their immutable receipts together before claiming a release is live. The [October 2026 promotion record](IMAGINE-BASELINE-PROMOTION.md) pins the completed 27B rollout.

## 1. Select and verify the training export

Start from a completed Runpod training model bundle and its original prepared manifest. Verify the bundle SHA-256, byte length, selected epoch, base-model revision, dataset release and revision, ontology snapshot, code commit, prompt, schema, processor, and `uv.lock`. The selected NF4-plus-adapter result is a **training configuration**, not the score of a merged or GGUF export. The [A40 run record](RUNPOD-FULL-A40-20261002.md) identifies the selected epoch and its original validation and separate final-assessment reports.

Preserve the atomic-task split. Use the same held-out validation image IDs, bytes, gold labels, rendering, prompt, schema, and ontology for each later diagnostic. Do not use the final-assessment cohort to tune export or serving choices. The evaluation contract is `edugraph_classify.evaluation.evaluate`: exact-set match is primary, with explicit and canonicalized precision/recall/F1, dimensions, views, training-frequency slices, invalid outputs, latency and token use. It does not score ontology-derived closure as an explicit prediction.

## 2. Benchmark material changes on the same cohort

Use the optional [Runpod inference benchmark](RUNPOD-INFERENCE-BENCHMARK.md) when the selected adapter, base revision, prompt/schema, export recipe, serving runtime, or target GPU changes substantially. `inference-benchmark prepare` copies the original validation cohort and training-frequency reference into one hash-pinned input bundle. The route selects only the model and serving stage:

| Stage | Route and output | What the comparison can establish |
|---|---|---|
| Selected raw training checkpoint | NF4 plus adapter; saved `epoch-<selected>-predictions.json` and metrics in the verified model bundle | Original validation baseline for this model and contract |
| Merged checkpoint | `merged_bf16_hf`; BF16 Transformers report, merged-shard hashes, no candidate model upload | Quality after BF16 base loading and adapter merge; precision changed too |
| Quantized serving candidate | `gguf_q4_k_m`; Q4_K_M language GGUF, BF16 projector, llama.cpp report and model hashes | End-to-end candidate quality, GPU fit and runtime on its tested hardware |

All three stages use the same evaluator. The offline `inference-benchmark compare` command additionally replays saved raw predictions; checks the source bundle's training manifest, selected result, reports and copied prompt/schema/template/processor against the original training manifest; and requires each inference manifest to retain those same control-file and validation-image hashes. It also requires identical validation rows and training-frequency references, checks reported metrics and prediction image hashes, and pairs exact-set outcomes by task ID at concurrency one. Supply at most one BF16 report and one GGUF report per invocation; duplicate routes are rejected. When both are present, it verifies identical merged-checkpoint file hashes and reports **every BF16-repeat × GGUF-repeat pairing**, with the repeat numbers. In that cross-route section, `regressed` means BF16-correct/GGUF-wrong and `recovered` means BF16-wrong/GGUF-correct. These paired counts describe end-to-end route differences, not isolated quantization loss. Verify the selected source bundle hash against the immutable training receipt, and each downloaded benchmark report archive against its completion-marker SHA-256, before running this command; the comparator's local cross-checks do not replace either external pin.

```powershell
uv run --no-sync edugraph-classify inference-benchmark compare `
  --training-manifest runs/<training-run>/manifest.json `
  --source-model-bundle artifacts/<verified-selected-model>.tar `
  --selected-epoch <epoch> `
  --training-predictions reports/<selected-export>/reports/epoch-<epoch>-predictions.json `
  --training-metrics reports/<selected-export>/reports/epoch-<epoch>-metrics.json `
  --benchmark-manifest runs/<gguf-run>/manifest.json `
  --benchmark-result reports/<gguf-run>/result.json `
  --benchmark-manifest runs/<bf16-run>/manifest.json `
  --benchmark-result reports/<bf16-run>/result.json `
  --output reports/<comparison>/paired-validation.json
```

Manifest/result options are paired in order; one benchmark pair is sufficient when only one route was run. The command is local and makes no provider call. For the released 27B candidate, the paired original-validation results are **43/64 NF4**, **40/64 merged BF16**, and **37/64 corrected GGUF**. Both BF16 and GGUF have two concurrency-one repeats, so the cross-route section contains four pairings; the first has 35 cases correct in both, five BF16-only, two GGUF-only and 22 wrong in both. These differences cannot be attributed to quantization alone because precision, conversion, runtime, and JSON constraint mechanics also change. The [GGUF](RUNPOD-GGUF-BENCHMARK-20261004.md) and [BF16](RUNPOD-BF16-BENCHMARK-20261005.md) records preserve the original evidence. Runpod speed and fit are not Cloud Run speed or fit.

## 3. Publish the serving artifact

Choose a successful GGUF candidate only after reviewing its paired quality and raw failure cases. Verify the whole candidate archive and every released member. Check the exact upstream weight license and attribution; this project's code license alone does not license a derived checkpoint. The [Hugging Face release workflow](HUGGINGFACE-GGUF-RELEASE.md) stages only the Q4_K_M language GGUF, matching BF16 projector, chat template, prompt, closed schema, upstream license, model card, and sanitized provenance. Publish the curated directory under the EduGraph-first model name, read back its immutable Hub revision and all file hashes, and retain the [publication receipt](../experiments/hf-gguf-publication-qwen38-27b-v1.json). Do not place a mutable Hub branch in a production image recipe.

Build the GCP llama.cpp image from that pinned Hub revision. The build verifies model, projector, and template bytes and records the resulting image digest. Test the exact digest as a private, zero-normal-traffic predictor revision before the application uses it. The [27B predictor canary receipt](../experiments/imagine-gguf-gcp-canary-20261006.json) records the actual L4 load, schema response, memory and cold-start smoke; neither a Runpod benchmark nor a successful build substitutes for this check.

## 4. Promote with tagged blue-green routes

Keep the old predictor and backend revisions as blue defaults while creating green revisions with stable, zero-normal-traffic Cloud Run tags. Configure the green backend to call the **tagged private predictor** using the service root as its OIDC audience. Smoke the predictor, then the backend root, ontology and disabled-search contract, a pinned validation upload, and the unchanged built-in large-image upload. The current release uses `qwen38-canary` for `edugraph-predict-00006-nej` and `qwen38-classify-v2` for `imagine-server-qwen38-40135eb`; their tagged URLs receive intended requests while their services' normal traffic remains 0%. The [backend v2 canary receipt](../experiments/imagine-backend-classification-canary-v2-20261006.json) records those checks.

Build the classification-only web bundle against the **green backend tag URL**, run a hosted preview with browser uploads, then publish the identical bundle to Firebase Hosting. Verify the live asset hash and upload through the custom domain. The [live release receipt](../experiments/imagine-classification-only-live-release-20261007.json) pins the web version, both green revisions, model revision and smoke results. This web release is the effective traffic switch for new page loads. Old open tabs still call the blue backend's legacy `/classify_and_search` route; switching either Cloud Run default to green today would break those tabs. Keep both blue defaults and their images available until a separately tested compatibility or retirement plan exists. Do not remove a green tag while a live web bundle references it.

Rollback begins with the prior Firebase Hosting version, then verifies the blue backend and predictor. Keep the prior web version, default revisions, image digests, and release receipt until rollback is no longer required. Blue-green behavior, tag audience, smoke checks and rollback are detailed in `imagine-server/docs/MODEL-CONTRACT-MIGRATION.md` and `imagine-web/docs/CLASSIFICATION-ONLY-CANARY.md` in their respective repositories.

## 5. Close the run without losing provenance

After durable completion or failure markers and report/candidate checks, terminate or remove the Runpod Pod and reconcile its billing state. Remove temporary scoped access grants and hosted preview channels after their checks. Keep immutable source bundles, benchmark reports and candidates referenced by published receipts; keep the Hub revision, production image digest, green tags, and blue rollback resources. Ignored local build contexts, downloaded duplicate artifacts, transient logs and preview bundles can be removed after hashes and receipts have been verified. Record any retained resource and the condition for its eventual removal. Every paid build, upload, Pod, or deployment requires the explicit provider-cost confirmation in `AGENTS.md`.
