# Secure GPU timing and recovery benchmark

Status: **Secure L40S benchmark completed; Pod automatically terminated and ongoing Runpod spend verified at zero.** Secure RTX 6000 Ada capacity remained unavailable, so the user explicitly selected L40S. Pod `xka533lhiit4u1` was allocated at 18:33:11 UTC on 2026-10-02, at $1.09/hour GPU / $1.114/hour with running disk, with 125 GB host RAM and 16 vCPUs in `US-MO-1`. It used a one-hour worker watchdog and $1.20/hour GPU allocation guard. All 277 local tests passed with 96.30% production-code coverage. The subsequent [full A40 training run](RUNPOD-FULL-A40-20261002.md) is recorded separately.

Allocation to durable result publication took **23.47 minutes**. The observed Runpod account balance decreased by **$0.44024** after termination; this excludes GCS and tax and is not an invoice. The independently verified summary is retained at `gs://edugraph-classify/runpod/benchmarks/edugraph-20261002-qwen38-27b-l40s-benchmark-v1/verified-summary.json`.

The original authorized target was one Secure RTX 6000 Ada 48 GB, with at least 48 GB host RAM / 8 vCPUs, the existing 170 GB disk allowance, a $1/hour GPU allocation guard and a one-hour worker watchdog. At its observed $0.84/hour GPU price, the half-hour planning estimate was $0.432 including running disk. The explicit L40S substitution changed the GPU and price settings. Startup precedes the worker watchdog and these are operational limits, not guaranteed billing caps.

## Observed result and decision

| Matched workload | A40 | L40S | L40S throughput gain |
|---|---:|---:|---:|
| Optimizer step, 18 warm paired batches | 15.023 s | 11.404 s | 1.317× |
| Prediction, 20 paired images | 10.996 s | 8.007 s | 1.373× |
| Total generated tokens in those predictions | 999 | 991 | Similar output volume |

The same initial adapter, data order, model revision and software were used for training. Generation restored the same trained A40 checkpoint. Twelve of twenty generated texts were identical across devices; the other eight remain preserved as raw outputs. The largest absolute difference between paired replay losses was 0.01214. This establishes useful throughput and recovery evidence, not bitwise cross-device identity or improved classification quality. Exact-set match still governs model selection in a full run.

Model initialization, including its download, took **11.56 minutes**, with **14.70 minutes from allocation to model-ready**. Generation, training and startup are reported separately so this cold-start delay does not distort the compute comparison. The cause of the additional loading time was not isolated. The comparison also includes differences in host CPU, memory and I/O.

Applying the measured speedups to the full two-epoch workload (828 steps and 490 generated classifications), with the same 0.35-hour overhead allowance, gives **4.01–4.14 hours / $4.47–$4.61 for L40S**, versus **5.24–5.40 hours / $2.69–$2.78 for A40**. L40S saves about **1.2–1.3 hours for $1.78–$1.84 extra**. For scheduling, allow **4–5 hours / $4.46–$5.57 on L40S**, compared with **5–6 hours / $2.57–$3.08 on A40**. These are extrapolations, excluding GCS, taxes and retries; slower cold starts or checkpoint transfers can exceed the common overhead allowance. The prediction ratio comes from the trained checkpoint and is also applied to base-model evaluation, which remains an assumption.

L40S is a reasonable choice when saving roughly an hour is worth about two dollars. A40 remains the measured cost-efficient choice. This workload obtained a modest speedup despite the faster hardware; advertised TFLOPS alone would have overstated the gain. Ada remains unmeasured and no further allocation was launched.

## Recovery and memory

The worker restored all **992 finite optimizer states at step 40**, restored RNG again after generation, and executed the actual first batch of epoch two. All optimizer states advanced to **41** with finite tensors. The saved continuation is epoch 1 / offset 4 / step 41, with the previous selected adapter kept separately. An independent GCS download verified its SHA-256, all 992 language-only adapter tensors, all 992 finite optimizer states at step 41, RNG presence and resume compatibility. The timing updates and later memory-only update are excluded from it.

The longest question and solution inputs (1,281 and 1,302 training tokens) completed an optimizer update without truncation. Their probe took 19.30 seconds. The longest prompt (1,260 tokens) then generated a normal 30-token response in 6.15 seconds using the restored trained checkpoint. It allowed up to 512 output tokens but did not exercise a forced 512-token answer.

| Memory measurement | GiB |
|---|---:|
| Peak PyTorch tensor allocation | 29.949 |
| Peak allocator reservation | 36.531 |
| Device memory in use at probe completion | 37.051 |
| Usable device capacity | 44.392 |

Retain a nominal **48 GB GPU** for this recipe. The earlier approximately 29 GiB W&B tensor peak omitted allocator reservation and other device memory. These probes establish fit on this L40S; they do not establish fit on a 32 GB device or the worst-case 512-token generation. The live audit again found **58,363,904 trainable language-adapter parameters and zero trainable vision or bridge parameters**.

## Evidence

- [W&B benchmark run](https://wandb.ai/edugraph-io/edugraph-classify/runs/edugraph-20261002-qwen38-27b-l40s-benchmark-v1).
- Run ID: `edugraph-20261002-qwen38-27b-l40s-benchmark-v1`.
- Original image: `ghcr.io/christian-bick/edugraph-classify-trainer@sha256:d77c8fffdeeb6c0091b7b58a44174b278d48ddac60f9ffa1955d1f2ab1843e64`.
- Report bundle SHA-256: `8843420bb72469b2a8a133d55c22fc2ae1cd413f1bda0a4b169ede26fd818641`, 194,560 bytes, under `gs://edugraph-classify/runpod/benchmarks/<run-id>/`.
- Continuation SHA-256: `8027421c178c6874fa6103509e558e77302f20dc70e4dffadc10a0794cdb3c89`, 935,424,000 bytes, under `gs://edugraph-classify/runpod/checkpoints/<run-id>/`.
- Small immutable completion record: `gs://edugraph-classify/runpod/runs/<run-id>/completed.json`.
- Independent downloaded report, paired analysis and recovery verification: `reports/runpod-hardware-20261002/<run-id>/` (ignored). The paired timing calculation was independently reproduced from the hash-verified GCS report.

The benchmark reuses the exact A40 image digest and trainer commit `7e51615ca0f6edfd548918dfd2b87c180fd6d258`. A separately committed and SHA-256-verified [experiment harness](../experiments/benchmarks/runpod_hardware.py) replaces only the orchestration entrypoint. It calls the existing worker with a diagnostic executor, retaining its Secure allocation, price, bundle, image-code and lockfile checks, GCS/W&B integration, watchdog, failure stop and publication-before-termination behavior. The harness is fetched from its immutable public Git commit and verified before execution. The prepared bundle separately records that harness commit/hash and the diagnostic specification hash. No account-wide Runpod credential is injected.

The [local preparation script](../experiments/benchmarks/prepare_runpod_hardware.py) verifies the original smoke artifacts, refuses changed trainer sources/dependencies, and reuses their original cohort and runtime identity. Additional memory-probe images are independently hash-verified from the full dataset audit. They are separate from the resumed training inputs so checkpoint compatibility is unchanged.

## Measurements

1. Restore the external step-zero A40 checkpoint and replay exactly its first 20 optimizer batches. Compare steps 3–20 with the matching A40 steps; the first two are warm-up on both sides. Preserve losses and batch identifiers as well as timing.
2. Restore the external epoch-one/step-40 checkpoint, including optimizer and RNG state. Generate two warm-ups followed by 20 fixed held-out classifications already present in the A40 export. Compare paired latency, output token counts and exact text agreement. Hardware-dependent output differences remain visible.
3. Restore step 40 again to remove sampling's RNG effects, process the actual next epoch-two batch, check all optimizer states advance to 41, and publish a coherent continuation checkpoint at epoch 1 / offset 4 / step 41. This is an external-checkpoint restore onto a new GPU Pod. It is not a complete second epoch.
4. Run one discarded training update on the longest question and solution training examples, repeated to fill the effective batch of four; generate normally from the longest prompt using the original trained checkpoint. Record allocated, reserved and device-used VRAM. Normal generation uses the 512-token ceiling but does not force an answer to consume all 512 tokens. No final-assessment example is used.
5. Publish the report and flush W&B, then terminate the Pod. The memory-probe update is excluded from the saved continuation. The diagnostic never promotes a model or publishes a new selected model.

The two-epoch recipe ceiling exists solely to validate continuation from epoch one; the harness always executes the bounded steps above. It records 22 optimizer updates across independent branches, not 22 sequential continuation steps. Timing and recovery branches are separate. W&B uses an increasing diagnostic event index, with the true optimizer position in each event's fields.

The actual L40S run is `edugraph-20261002-qwen38-27b-l40s-benchmark-v1`. It uses the identical harness/specification/checkpoints as the prepared Ada variant; only the run ID, GPU and price settings differ. The local preparation command now supports `--gpu l40s` to reproduce that choice. Its input bundle is `gs://edugraph-classify/runpod/inputs/edugraph-20261002-qwen38-27b-l40s-benchmark-v1/9ea55b5fb92e2afc884670909e570bb0660e8a38128147830b9e62c5e673948d.tar`. The half-hour estimate is $0.557. Comparing complete allocations includes differences in host CPU, memory and I/O; it does not isolate GPU silicon performance.

The benchmark tests use fake providers and a tiny checkpoint engine to verify bounded execution, exact batch order, step-40 restoration, step-41 publication, exclusion of memory-only updates, finite optimizer checks, and failure behavior. The original trainer and production inference configuration remain unchanged.

## Prepared Ada run

- Run ID: `edugraph-20261002-qwen38-27b-ada-benchmark-v1`.
- Harness commit: `c2050ec3676feec1ee52759438373209807065b8`; source SHA-256 `ad78bf082118a542a87965a324949bc9c72fc4c55cbb1c90b84630be832686a7`.
- Input bundle: `gs://edugraph-classify/runpod/inputs/edugraph-20261002-qwen38-27b-ada-benchmark-v1/6475ca8c85bc6e14eb01508c7a5b4c9f374960935e43ceb840fd37f032d831e3.tar` (30,423,040 bytes).
- The 18 matched A40 training steps average 15.022954 seconds. The 20 matched A40 predictions average 10.995955 seconds and contain 999 output tokens in total.
- Memory probes have 1,281 and 1,302 training tokens; the largest prompt plus the full generation allowance is 1,772 tokens.
- Local manifest, specification, rendered request and capacity observations: `runs/secure-ada-20261002/edugraph-20261002-qwen38-27b-ada-benchmark-v1/` (ignored).

No Ada throughput result exists. The preparation verified the public harness download/hash, original trainer sources, external checkpoint compatibility, image bytes and rendered inputs locally. Running the updated preparation command with `--gpu l40s` reproduced the actual L40S manifest byte for byte.
