# Imagine K–6 production release — 2026-10-26

The published K–6 classifier is live at [demo.edugraph.io](https://demo.edugraph.io). The public API and private L4 predictor both serve the verified release by default. The [machine-readable receipt](../experiments/imagine-k6-promotion-20261026.json) binds source commits, model and application contracts, provider identities, test outcomes, cleanup and rollback state.

## Release identity

| Component | Pin |
|---|---|
| Model | `christian-bick/EduGraph-Classifier-Qwen3.8-27B-GGUF` at `140480b1d3893f0fb678a6f2d0f92103a60a936b` |
| Dataset / ontology | `v0.31.0-01` / `0.31.0` |
| Classifier build source | `e900b4ef2b2e061c215466ede809d8ee27ac7c54` |
| Backend image source | `27374a79d8db6d80ef5c6124f47160183eba5aa8` |
| Web build source | `444c81d1e820b4cd26e3e501dc353d2d8d2dafca` |
| Predictor image | `sha256:f3077e15d5a04aea3c51deea341384592d6e407242f5d9c59478e7ce8d59027d` |
| Backend image | `sha256:a752811efb320192eb0047f911a38a78f30333d0360905db251d934f6b7f7409` |
| Predictor / backend defaults | `edugraph-predict-k6-140480b1` / `imagine-server-k6-default-27374a7` |
| Hosting version / release | `1df2038228449e23` / `1792976793555000` |
| Hosted JavaScript SHA-256 | `85bb85d29c1f00eee1fc2a4ad99a43037b541a3dbd9caf962c8069c8db4e37e5` |

The language model remains Q4_K_M with a BF16 projector and the pinned llama.cpp `0faee5004297c3bcfa40b7bf11750127b8c1fd7d` runtime. Context, logical batch and microbatch are 4,096, with greedy generation, 512 output tokens and concurrency one. Cloud Run retains one L4, four CPUs, 16 GiB host memory, a one-instance cap and scale-to-zero. The predictor remains private; the public backend authenticates with its existing service account.

Backend profile `qwen38_gguf_k6_v1` is paired with `edugraph_v031`. The published ontology RDF, prompt and closed schema are hash-pinned. The full ontology has 894 Areas, 417 Scopes and 130 Abilities; the output schema's eligible vocabulary is 768, 389 and 117. RGB JPEG quality 75 preserves native images within 4,096 pixels per side and 3,145,728 total pixels; larger inputs resize proportionally. Predictions remain explicit, and search endpoints remain deliberately disabled. The web cache key changed to `classification-storage-v031-140480b1`; an already-open older page needs a reload to acquire the new bundle and ontology.

## Verification and rollout

Local verification passed: classifier 410 tests with 95.83% coverage, backend 98 tests, web lint and 22 unit tests, the delayed mock-upload/reload flow and production build. The exact backend image also passed a non-root, read-only, network-disabled container smoke. Linux backend CI [run 40248820768](https://github.com/christian-bick/imagine-server/actions/runs/40248820768) passed on the pinned source commit.

Cloud Build `586721e3-39b5-4c9a-982c-72fb5ae4a379` succeeded after 33m28s, including approximately 20 minutes publishing the large weight layer. Artifact Registry independently resolved the expected digest, runtime revision and command. The Cloud Run deployment operation reported a readiness-deadline error, but the same revision subsequently became healthy: image import took 13m46s and container health took 20s. Direct revision, service and tag readback confirmed success at generation 10. No replacement revision or retry was created, and traffic stayed on the old model throughout preparation.

Before changing the predictor default, the old backend was cloned with its original immutable image and pinned to old predictor tag `qwen38-canary`. Its tag and public default passed the existing smoke. This established a compatible rollback pair across the ontology change. The new private predictor and tagged K–6 backend were then tested at zero default traffic.

| Private L4 canary case | Prompt tokens | Client time | Check |
|---|---:|---:|---|
| Pinned angle validation image | 465 | 4.645 s | Exact benchmark labels |
| Grade 6 statistical question, longest selection image | 1,361 | 5.029 s | Exact benchmark labels |
| Longest training-corpus image, 800×3,362 | 2,738 | 10.773 s | Completed, valid closed-schema output |
| Built-in addition example, 1,495×1,920 | 2,933 | 12.815 s | Completed, valid output |
| Synthetic upload boundary, 4,096×768 | 3,185 | 10.562 s | Completed, empty valid label sets |

All five returned HTTP 200 and `finish_reason=stop`. The longest-image and boundary checks establish fit and contract validity, not label accuracy. These requests ran after deployment startup and may benefit from caching; they do not measure a cold-start distribution.

The tagged backend passed full ontology/taxonomy, disabled-search and exact pinned-classification checks, plus Grade 6, longest-image and built-in uploads. ETag-guarded traffic changes promoted the predictor first. Its default was verified, then a new backend calling that untagged predictor was deployed, smoked and promoted. The public API passed the full contract check.

Hosted Chrome checks on both preview and live verified the complete pinned ontology, built-in addition upload, exact Grade 6 statistical-question labels, results route, visible labels, absence of search controls and no browser errors. Firebase released the exact preview version without a rebuild; independent custom-domain readback matched the JavaScript hash. A subsequent two-request warm probe returned HTTP 200 twice in 4.946 and 9.049 seconds. This is a bounded burst check, not a sustained-load guarantee.

## Observation, cleanup and rollback

Monitoring for **00:59–01:08 UTC**, collected after ingestion delay, recorded all 18 predictor requests as HTTP 200 and no revision-scoped allocation/error logs. Sampled GPU memory peaked at **19,061,014,528 bytes (17.75 GiB; 78.92%)** and predictor host memory at **4,196,237,312 bytes (3.91 GiB)**. The observed maximum was one predictor instance and one active request. Metrics sample every 60 seconds and may arrive up to 120 seconds later, so these are sampled peaks rather than continuous maxima. All three revision-scoped result sets were untruncated.

The earlier backend canary recorded seven HTTP 200 responses and two deliberate HTTP 503 responses; the canonical backend recorded 18 HTTP 200 responses and four deliberate HTTP 503 responses. Sanitized request paths confirmed every 503 was `/search` or `/classify_and_search`, as required by the disabled-search checks. No other backend HTTP error was observed in this bounded window. Final readback confirmed the public backend and private predictor IAM boundaries and both K–6 defaults.

GCP build/deployment/inference billing was not available as a finalized charge during verification. No deployment-cost total is claimed; prior RunPod training/conversion costs are separate.

After live verification, the temporary OIDC smoke grant was removed, preview channel `k6-model-20261026` was deleted with absence confirmed, and unused backend tag `model27b-default-v1` was removed. That old backend called the changing default predictor and was not a compatible old-model rollback. The current and preceding web bundles were verified to call only the untagged public API. Independent final inventory confirmed the new routes and live version after cleanup.

The compatible old pair remains: backend `imagine-server-k6-rollback27b` at tag `k6-rollback27b`, calling predictor `edugraph-predict-00006-nej` through `qwen38-canary`. Hosting version `5478bb45b1ea67b5` is retained with that pair through **2026-10-27 01:15 UTC**, a 24-hour rollback window rounded up from verification. No retirement job is scheduled; recheck dependencies and traffic before removing those resources. To roll back, restore the paired backend and predictor defaults with guarded traffic changes, release the retained web version and repeat the pinned API/browser checks. Preserve historical experiment receipts.

## Quality evidence and limits

The [matched GGUF benchmark](RUNPOD-GGUF-K6-20261025.md) remains **43/64 exact-set match (67.19%)**, precision **95.47%**, recall **95.29%**, micro F1 **95.38%**, and zero invalid outputs in both repeats. Its matched NF4 source scored 47/64; no standalone BF16 diagnostic isolates the export/runtime difference. The separate 306/414 NF4 assessment is not a GGUF score, and the GGUF final cohort remains unmeasured.

The corpus covers in-scope CCSS Mathematics from Kindergarten through Grade 6; [coverage and exclusions](https://coverage.edugraph.io) describe that scope. Official validation omits 147 of 403 training labels, and 224 assessment images reuse previously inspected bytes. Label conventions, completeness and fresh evaluation remain deferred. This rollout establishes working production integration and tested input fit, not mastery of every standard or a new unbiased quality estimate.
