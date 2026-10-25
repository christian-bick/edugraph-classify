# K–6 GGUF conversion: startup failure, 2026-10-25

The approved conversion and validation attempt stopped before any observed worker execution. **No verified K–6 GGUF candidate or evaluation result is available, and Hugging Face was not updated.** The [sanitized receipt](../experiments/runpod-k6-gguf-startup-failure-20261025.json) records the provider state, timestamps, budget and publication check.

## Prepared release

The [export recipe](../experiments/inference-benchmark-qwen38-27b-k6-gguf-v1.json) selects epoch 2 from the [verified K–6 training run](RUNPOD-K6-A40-20261024.md), dataset `v0.31.0-01` and ontology `0.31.0`. It preserves the established BF16 merge, Q4_K_M language quantization and BF16 projector route. It requests two repeats on the same 64 selection images with context, logical batch and microbatch each set to 4,096. The separate 414-image NF4 assessment is not a GGUF result.

Execution code `7c9db04e3bc3cf9b018bf8912bac82ddc680c29e` passed 378 tests with 95.83% coverage and `uv lock --check` before launch. The matching [image build](https://github.com/christian-bick/edugraph-classify/actions/runs/40191673855) succeeded. Anonymous registry checks verified its manifest/config digests, Linux amd64 platform, code and llama.cpp revisions, and worker entrypoint:

`ghcr.io/christian-bick/edugraph-classify-inference-benchmark@sha256:2570eea4b136be1727888e17097bfacf828b42c32cf361b7a451d3a4f9cb46be`

The staged input archive is 26,490,880 bytes, SHA-256 `234fc36fd33bde0b2355022fcb61d8a1b63023ea58a0bc3ca86865d27bd045f6`. The selected adapter/source archive remains verified at SHA-256 `57e381ec0a7026e70775beca47e9732554fc50562533a78bb7b8f33cffe04e6a`. No execution code or recipe changed during the allocation.

## Allocation and recovery

Run `edugraph-20261025-qwen38-27b-k6-gguf-v1` used one Secure L40S Pod, `s9nbl9q3cku2ug`, with 125 GB host RAM, 32 vCPUs, a 30 GB container disk and a 350 GB persistent volume. The provider reported $1.09/hour compute.

| Event | UTC time on October 25 |
|---|---|
| Allocation budget began | 12:53:46 |
| Provider created Pod | 12:53:53 |
| One same-Pod restart accepted | 13:21:05 |
| Last runtime check: still null | 13:31:19 |
| Last marker check: startup, failure and completion absent | 13:31:24 |
| Provider recorded explicit stop | 13:31:39 |
| External watchdog independently observed `EXITED` and exited | 13:31:58 |
| Independent stopped-state and unchanged Hub verification | 13:32:17 |

The original budget included startup: an external stop-only watchdog was scheduled for 15:48:46, five minutes before the absolute three-hour deadline of 15:53:46. The recovery restart did not extend that budget. It was journaled before the single provider request; the provider returned HTTP 200 and updated its restart timestamp. No replacement Pod was created.

Throughout observation, the provider's desired state was `RUNNING`, but GraphQL reported no container runtime. Fresh bounded system/container log reads returned HTTP 200 with no events. GCS reads succeeded but found no startup, worker-failure or completion marker. The same state persisted for ten minutes after the restart, so compute was stopped rather than exhausting the remaining budget. The provider then reported `EXITED`; the Pod was retained, not terminated.

These observations locate the failure before any confirmed worker execution. They do **not** identify whether image pulling, container creation or host startup failed. The public image was accessible, but that does not prove the host could pull it. No model-loading, memory, conversion or quality failure was observed. The worker's own deadline could not substitute for the external startup-inclusive cutoff.

## Cost and retained state

The creation-to-stop interval was 2,265.559 seconds (37 minutes 45.559 seconds). Applying the reported compute rate gives approximately **$0.686**. Including the recipe's $0.05/hour running-disk allowance gives approximately **$0.717**. This is a rate-based estimate, not an invoice; it excludes GCS, taxes and storage after stopping. Provider billing may differ, including any unavailable-host treatment.

The 350 GB volume is retained for recovery. [Runpod's published pricing](https://docs.runpod.io/pods/pricing), checked October 25, lists stopped volume storage at $0.20/GB/month: **$70/month equivalent while retained**, billed per second, with no charge when the host is unavailable. GPU compute has stopped. Deleting the Pod would remove that retained volume; no deletion was performed.

Local launch, deadline, restart, runtime and shutdown records remain under the ignored run/report directories. Prepared inputs and the source model remain in durable GCS storage. There is no verified candidate or report to retrieve from this attempt.

## Publication and next step

Anonymous readback confirmed that `christian-bick/EduGraph-Classifier-Qwen3.8-27B-GGUF` still points to `477e3caf32eced2a9e002ead1edccc9a5c458f0c`. That revision contains the previous model. Its weights and README were not relabeled as K–6, and no Imagine/GCP deployment changed.

The remaining work is a successful conversion allocation, paired validation, artifact verification and one guarded Hub publication. Resolve the provider startup problem or review a fresh allocation before retrying. Preserve this failed run's identity and journal; do not reuse its launch record for a second Pod or extend the original deadline silently.

The eventual card should describe **in-scope CCSS Mathematics Kindergarten–Grade 6 training-corpus coverage**, linking the [Coverage Explorer](https://coverage.edugraph.io). The accepted limitations still apply: 147 of 403 training labels are absent from official validation, and 224 images across selection and final validation reuse historical assessment images. The training results are regression evidence, not fresh blind evidence or verified mastery of every standard.
