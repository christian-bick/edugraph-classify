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

Deploy the new private predictor and classification backend as digest-pinned Cloud Run revisions with unique tags and zero **default** traffic. Keep their service-account/IAM boundary unchanged. The candidate backend initially calls the tagged predictor URL, with the untagged predictor service root as its OIDC audience. Smoke that tag with a hash-pinned benchmark image, the built-in large image, `/ontology`, `/taxonomy`, and the disabled-search contract. A smoke establishes request compatibility, not the 64-case quality result. The [October 2026 backend canary receipt](../experiments/imagine-backend-classification-canary-v2-20261006.json) records the first 27B integration check.

For a default-route cutover, first read the live revision IDs, traffic, ETags, image digests, scaling and current web bundle URL. Add temporary rollback tags to the old backend and predictor. If old clients use an incompatible endpoint, record the decision to retire that contract explicitly; do not assert compatibility to bypass a traffic guard. Promote the **backend default first**, then the predictor default: reversing the order would route old backend requests to a predictor with a different request schema. Verify the backend's untagged public API after each change. Keep the new tags until consumers stop calling them. `imagine-server/docs/BLUE-GREEN-RELEASE.md` maintains the exact traffic commands and rollback procedure.

Build a new backend revision whose `PREDICT_ENDPOINT` is the **untagged private predictor** `/v1/chat/completions` URL, and whose OIDC audience remains its untagged service root. Test that revision under a zero-default-traffic tag, then promote it with an ETag-guarded traffic update. This removes the backend's dependence on a temporary predictor tag while retaining a tested 27B revision for rollback. Keep predictor access private; the public backend performs the authenticated call.

Build the web bundle against the **untagged public backend** `https://api.edugraph.io`. Publish a non-live Firebase Hosting preview using the web repository's hash-checked Hosting REST publisher, which avoids the Firebase CLI's Service Usage preflight, and test it with the built-in and pinned validation images. Record the preview version and JavaScript hash. Release that **exact tested version** to live with a guarded Hosting REST release-by-version call, checking both the expected preview and previous live version before the write and reading back the resulting live version. `firebase hosting:clone` is an optional equivalent when its credentials pass preflight. Verify the custom domain, route, labels and browser errors; retain the previous Hosting version and route tags only during the defined rollback window. The web repository's `docs/WEB-RELEASE.md` maintains the publisher and release commands. An old browser tab may still call a retired endpoint; that limitation is part of the incompatible-contract cutover, not a reason to leave 4B serving indefinitely.

## 5. Close the run without losing provenance

After durable completion or failure markers and report/candidate checks, terminate or remove the Runpod Pod and reconcile its billing state. Remove temporary scoped access grants and hosted preview channels after their checks. Keep immutable source bundles, benchmark reports, historical experiment records, and candidates referenced by published receipts; keep the Hub revision, production image digest, and a current-model canary with smoke tests. Once the new default backend, private predictor and live web have passed smoke and observation, remove obsolete traffic tags, old-model revisions, their unreferenced images and model-only storage after checking other consumers. Retiring 4B runtime resources does not erase the dated 4B training and migration evidence; preserve those records as history rather than as active instructions. Delete exact model prefixes, not shared buckets. Record the exact removed and retained resource IDs, post-deletion checks, and any storage soft-delete period. Ignored local build contexts, duplicate downloads and transient logs can be removed after their hashes and receipts are verified. Every paid build, upload, Pod, or deployment requires the explicit provider-cost confirmation in `AGENTS.md`.
