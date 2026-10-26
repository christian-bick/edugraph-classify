# Imagine classifier production baseline

As verified on **2026-10-26**, [Imagine](https://demo.edugraph.io) serves the **K–6 Qwen3.8-27B Q4_K_M GGUF** with its BF16 vision projector, dataset `v0.31.0-01` and ontology `0.31.0`. The training corpus covers in-scope CCSS Mathematics from Kindergarten through Grade 6; [coverage and exclusions](https://coverage.edugraph.io) describe corpus scope, not measured mastery of every standard. The [dated deployment record](IMAGINE-K6-PROMOTION-20261026.md) and [verified receipt](../experiments/imagine-k6-promotion-20261026.json) preserve the release evidence.

| Live component | Immutable identity |
|---|---|
| Model Hub revision | `140480b1d3893f0fb678a6f2d0f92103a60a936b` |
| Private predictor default | `edugraph-predict-k6-140480b1` |
| Predictor image digest | `sha256:f3077e15d5a04aea3c51deea341384592d6e407242f5d9c59478e7ce8d59027d` |
| Public backend default | `imagine-server-k6-default-27374a7` |
| Backend image digest | `sha256:a752811efb320192eb0047f911a38a78f30333d0360905db251d934f6b7f7409` |
| Firebase Hosting version | `1df2038228449e23` |
| Live Hosting release | `sites/imagine-web/releases/1792976793555000` |

The browser calls the untagged `https://api.edugraph.io` endpoint. The backend calls the untagged private predictor using its service account. Context, logical batch and microbatch are 4,096; concurrency is one, with one L4 and scale-to-zero. The ontology response contains 894 Areas, 417 Scopes and 130 Abilities. Explicit predictions remain separate from ontology closure; search remains unavailable.

The [matched K–6 GGUF benchmark](RUNPOD-GGUF-K6-20261025.md) scored **43/64 exact-set match (67.19%)**, **95.38% label micro-F1** and zero invalid outputs in both repeats, versus 47/64 for its NF4 source. The separate 306/414 final score belongs only to NF4; GGUF final-cohort accuracy is unmeasured. Assessment reuse and label coverage gaps remain, so these are regression measurements. Production smokes verified integration and bounded input fit, not a new accuracy benchmark or sustained-load capacity.

Private predictor, tagged and default backend, preview and live browser checks passed. The exact tested preview version was promoted without rebuilding. The temporary preview, temporary OIDC smoke grant and unused incompatible old backend tag were removed. The old-model rollback pair (`imagine-server-k6-rollback27b`, tag `k6-rollback27b`, calling old predictor tag `qwen38-canary`) and preceding Hosting version `5478bb45b1ea67b5` are retained through **2026-10-27 01:15 UTC** for a 24-hour rollback window. Retirement after that window requires a fresh dependency/traffic check; no automatic deletion is scheduled. A complete old-model rollback must restore the paired backend/predictor routes and old web version, then repeat smoke verification.

For subsequent releases, follow [Model to production](MODEL-TO-PRODUCTION.md) and the server/web blue-green runbooks. The [October 22 default cutover](../experiments/imagine-27b-default-cutover-20261022.json), [October 7 tagged rollout](IMAGINE-BASELINE-PROMOTION-20261007.md) and [cleanup record](GCP-CLEANUP.md) remain historical evidence.
