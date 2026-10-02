# Compact 27B training: token profile, runtime, and cost

Assessed 2026-10-02, then updated with the authorized [Secure A40 smoke](RUNPOD-SMOKE-20261002.md) and [L40S benchmark](RUNPOD-HARDWARE-BENCHMARK.md). The [full A40 run](RUNPOD-FULL-A40-20261002.md) subsequently launched at 21:27 UTC; its measured result is pending. The original Fireworks/A6000 comparison below used local rendering and read-only provider/catalog checks; the final sections add measured A40 and L40S throughput. Dollar values are USD, excluding tax, deployment, engineering time, and retries.

## Workload and measured inputs

The comparison covers Qwen3.8-27B, dataset v0.30.0-03 at `9b509867e898490615be3f59bc2f31fac429389c`, ontology v0.30.0, and prompt `direct-label` v3. The model/processor revision is `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It retains language-only rank-8/alpha-16 LoRA, assistant-only loss, and effective batch size four. Vision and bridge targets are excluded.

Two complete epochs mean **3,308 image presentations, 828 optimizer steps, and 490 generated classifications**: 80 for the base model, 80 after each epoch, and 250 on the final assessment cohort. Each 80-image phase contains 64 checkpoint-selection images and 16 training diagnostics. Generated predictions, exact-set match, F1, validity, frequency slices, and raw responses remain required. The tracked recipe retains the existing three-epoch ceiling and patience-one early stopping; this comparison deliberately shows the requested two-epoch case as well.

The local audit checked the immutable Hub file-tree hashes for all 1,968 images and both metadata files. It reused cached bytes only after verifying them against that tree, downloading the two corrected images. It found zero byte duplicates and no train/validation overlap. The real pinned tokenizer and processor rendered every example without truncation.

| Measurement | Previous vocabulary prompt | Compact v3 |
|---|---:|---:|
| System text tokens | 2,993 | 88 |
| Mean training tokens per image | 3,418.72 | 509.69 |
| Two-epoch training tokens | 11,309,128 | 1,686,068 |
| Maximum training length | Historical profile | 1,302 |
| Maximum prompt plus 512 output tokens, all cohorts | Historical profile | 1,772 |
| Local context guard | 16,384 | 2,048 |

The old column uses the prepared v0.30.0-02 full-v1 cohort; the current column uses corrected v0.30.0-03 pixels and shorter user wording as well as the compact system prompt. This is an approximately **85% reduction in training tokens**, not a controlled prompt-only ablation and not an 85% wall-time or total-VRAM reduction. Image processing, vision forward passes, optimizer steps, checkpointing, and generated validation still occur.

Audit evidence is in gitignored `reports/compact-training-20261002/`: `audit_compact.py`, `profile.json`, `examples.json`, `dataset-tree.json`, the closed decoding schema, and the verified embedded template. Prompt SHA-256: `5f1cd2c09349321a3ead765896b8e9b6c3afb403eb088c8135ca9d43f40335c4`. The profile records source and lock hashes; it is deliberately not a clean-commit launch manifest.

## Fireworks token cost

Live eligibility still reports READY, image input, LoRA, training V2, and supervised-LoRA tuning support. The Qwen3.8-27B **Serverless Training API** rates checked on 2026-10-02 are $4.103/M training tokens, $1.86/M uncached sampling input, and $5.595/M generated tokens. These differ from managed-SFT pricing. Training meters include the prompt even though its loss is masked. [Fireworks pricing](https://fireworks.ai/pricing), [meter definitions](https://docs.fireworks.ai/fine-tuning/training-api/serverless#what-the-meters-mean).

| Completed epochs | Optimizer steps | Training | Generated validation allowance | Total estimate |
|---|---:|---:|---:|---:|
| 1 | 414 | $3.46 | $1.53 | **$4.99** |
| 2 | 828 | $6.92 | $1.83 | **$8.75** |
| 3, recipe ceiling | 1,242 | $10.38 | $2.13 | **$12.50** |

For two epochs, the calculation is:

```text
training:   1,686,068 × $4.103 / 1,000,000 = $6.917937
prefill:      229,100 × $1.860 / 1,000,000 = $0.426126
generation:   250,880 × $5.595 / 1,000,000 = $1.403674
total:                                        $8.747737
```

Generation assumes all 490 responses consume the complete 512-token allowance and every prompt is uncached. Normal compact answers should cost less; this is a token-budget estimate, not a billing cap. The comparable old two-epoch estimate was $50.88, so the new estimate is approximately **83% lower**. An extra epoch plus its 80-image selection/diagnostic evaluation adds about **$3.76**, excluding another final assessment. A separately resumed run also needs its own restored-baseline evaluation; do not reuse the final assessment repeatedly during tuning.

The ontology enums are passed as a decoding constraint, without appending them to the prompt. Fireworks documents that the model does not automatically see the supplied schema. Runtime validation remains necessary for truncation, duplicates, and semantic errors; constraints cannot select the correct labels for an image. The exact image-completions/schema combination still needs a live compatibility check in the next authorized diagnostic. [Structured-output behavior](https://docs.fireworks.ai/structured-responses/structured-response-formatting).

## Runtime: planning assumptions, not measurements of the new run

**Fireworks: budget roughly 2–5 hours for two epochs including generated evaluation, with low confidence.** The earlier smoke processed 543,097 training tokens and 40 optimizer steps in a 33-minute elapsed interval that also included session setup, initial generation, and provider waits. It has no per-step timing breakdown. The compact full run has 3.10 times those training tokens but 20.7 times as many optimizer steps. Scaling solely by tokens gives about 1.7 hours; assigning one quarter of the old elapsed time to step-dependent work gives about 4.1 hours. The old 80-image base generation took roughly 190 seconds, implying about 19 minutes for 490 similarly sized generations. Rounded together, these scenarios produce the 2–5-hour allowance. The assumed overhead fraction is not measured, and shared-pool congestion or constraint compilation can exceed this range. Token cost does not increase merely because the shared pool is slow.

**Runpod: use a conservative 4–10-hour allowance on one RTX A6000 48 GB for the complete two-epoch QLoRA workload.** This is a conditional estimate: it treats the user's reported 1–2-hour private run as one epoch on two RTX 3090s, then allows roughly twice the elapsed training time on one comparable-generation GPU and doubles for two epochs. Setup, the slightly larger new compact prompt/output contract, and 490 generated validations add uncertainty; freezing vision adapters and improved batching may recover time. This is not a measured A6000 benchmark or a claimed throughput ratio.

The supplied private adapter confirms a 27B NF4/4-bit base and a short 42-token instruction, but does **not** record the final run arguments, epoch count, or wall-time log. Its README describes two-3090 DDP; that does not prove the final run used those arguments. If the user's 1–2 hours already covered two epochs, or the final run used only one 3090, a **2–6-hour allowance** on one A6000 is more appropriate. The prompt simplification makes this new workload closer to that older short-prompt run; it does not make it six times faster than that run. Measure actual step and generation throughput before making a tighter forecast.

## Runpod GPU choices and cost

New self-hosted containers use [GHCR](CONTAINER-IMAGES.md), whose container storage and bandwidth are currently free. GitHub Actions build minutes have separate billing; publication requires manual dispatch or an explicit trainer release tag. GCS run-artifact costs and Runpod disk/compute charges remain separate. The subsequent [GCP cleanup](GCP-CLEANUP.md) removed the legacy training registries while retaining production inference resources.

For a practical self-hosted option, start with **one RTX A6000 48 GB, NF4 QLoRA with BF16 compute, microbatch one and accumulation four**, gradient checkpointing, and exact language-only target paths. Keep image resolution and all examples. The pinned official base can be quantized locally; the imported private 4-bit model demonstrates feasibility but is not substituted silently as the new base. Reuse the v3 prompt, closed schema, canonical targets, cohort selection, and evaluator. The Runpod executor and local grammar decoder are now implemented. The [Secure A40 smoke](RUNPOD-SMOKE-20261002.md) completed 40 optimizer steps, verified zero trainable vision/bridge parameters, improved held-out label F1 and published its checkpoints and model. The measured A40 projection below is now the stronger runtime baseline; A6000 throughput remains unmeasured.

The following are on-demand **Pod** prices checked using both cloud selectors on the [Runpod pricing page](https://www.runpod.io/pricing), which showed an update date of 2026-09-27. They are list prices, not reserved availability. No spot discount is assumed.

| Configuration | Secure / hour | Community / hour | Role |
|---|---:|---:|---|
| 1 × RTX A6000, 48 GB | $0.53 | $0.33 | Primary economical QLoRA candidate |
| 1 × A40, 48 GB | $0.49 | $0.35 | Similar memory headroom; benchmark throughput |
| 1 × RTX 6000 Ada, 48 GB | $0.84 | $0.74 | Higher-cost performance candidate; speedup unmeasured |
| 1 × L40S, 48 GB | $1.09 | $0.79 | Another performance candidate |
| 1 × A100 PCIe, 80 GB | $1.59 | $1.19 | BF16-base LoRA candidate without 4-bit compression |
| 1 × H100 PCIe, 80 GB | $2.89 | $1.99 | Faster hardware candidate; not required just for fit |
| 1 × RTX 3090, 24 GB | $0.50 | $0.22 | Only after longest-example memory validation |

Using the conservative A6000 runtime allowance:

| Execution | Two-epoch elapsed allowance | Training + generated evaluation cost |
|---|---:|---:|
| Fireworks serverless LoRA | 2–5 h, low confidence | **$8.75** no-cache token estimate |
| Runpod Secure, 1 × A6000, QLoRA | 4–10 h, conditional | **$2.12–$5.30** GPU rental |
| Runpod Community, 1 × A6000, QLoRA | 4–10 h, conditional | **$1.32–$3.30** GPU rental |

If the 2–6-hour alternative applies, the Runpod totals are $1.06–$3.18 Secure and $0.66–$1.98 Community. At the same GPU throughput, cloud category changes the rate rather than inherently changing model compute time. For reference, an A6000 run would have to last approximately 16.5 hours Secure or 26.5 hours Community to reach the $8.75 Fireworks token estimate before storage. This makes the rental-cost conclusion less sensitive than the runtime forecast.

Runpod charges for loading, generation, checkpointing, and idle debugging while the Pod remains running. At $0.10/GB/month, 100 GB of running disk adds approximately $0.06–$0.14 over 4–10 hours (using a 720-hour month); retained storage continues to cost money afterward. More disk is needed if keeping both BF16 and quantized base copies. A40/A6000 listings include about 50 GB host RAM, so BF16 loading must stream/shard rather than build a complete CPU copy plus duplicates. Other GPU types can offer more host RAM. [Runpod pricing](https://www.runpod.io/pricing).

The public dataset/model made Community Cloud acceptable for the earlier comparison; the user subsequently selected **Secure** because of the credential trust boundary. Host reliability and disk/network variability also differ: Secure uses data centers with higher redundancy, while Community uses peer providers. Preserve resumable checkpoints outside an ephemeral container and avoid treating advertised availability as a guarantee. [Runpod Pod selection](https://docs.runpod.io/pods/choose-a-pod#secure-cloud-vs-community-cloud).

## What the shorter prompt changes about VRAM

Shorter sequences reduce language activations during training and the attention cache during generation. They do not shrink the base weights, image patch count, or the vision encoder. The 2,048 local context guard fits this complete release, but the current Fireworks SDK does not forward it as a pool-memory reservation; it is not an eightfold provider-VRAM saving.

The inspected official BF16 base contains about **55.6 GB of weights** before activations, adapter/optimizer state, and workspace. It cannot fit entirely in 48 GB through prompt shortening alone. **80 GB is a reasonable BF16 LoRA candidate**, subject to measured peak memory. The inspected private NF4 checkpoint contains about **18.6 GB of weight files**, which makes **48 GB a reasonable QLoRA target with substantial additional headroom**, rather than proving its exact GPU allocation. QLoRA changes base precision and must be evaluated as a distinct numerical training variant from the provider-managed run; Fireworks' exact internal precision is not established by this audit. [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization).

Before the smoke, a 24 GB card was plausible but unverified. The old private pipeline excluded examples over 1,024 tokens and documented an out-of-memory case around 1,288 tokens. This new release has **31 training images over 1,024 tokens**, reaches 1,302 for training, and needs up to 1,772 including the generation allowance. Freezing vision may help, but no GPU memory trace proves that every case fits. Do not discard or truncate those images to claim success. Two 24 GB GPUs with DDP each hold a full model replica; they do not provide a single 48 GB allocation. A 32 GB card is another possible middle ground, but the installed CUDA/framework support and longest-example fit need checking before choosing newer hardware.

The later A40 smoke measured **29.122 GiB peak PyTorch allocation** with the implemented trainer. A 24 GB card therefore does not fit this implementation unchanged. The subsequent L40S longest-input probe passed at **29.949 GiB peak tensor allocation**, **36.531 GiB peak allocator reservation**, and **37.051 GiB device memory in use** at probe completion. A 32 GB device remains unverified; retain a nominal 48 GB device for the current candidate. The generation probe used a normal 30-token response, not the full 512-token allowance.

The user subsequently selected **Runpod Secure QLoRA** for its credential trust boundary. The [Runpod executor](RUNPOD-TRAINING.md) has W&B metrics, GCS checkpoints and recovery. Its two-epoch A6000 candidate uses the conservative 10-hour estimate at $0.53/hour plus a $0.024/hour running-disk allowance for 170 GB total: approximately **$5.54**, excluding GCS, W&B plan charges and tax. This remains conditional on capacity and measured throughput. The live 2026-10-02 query found no suitable Secure A6000, so the authorized 160-image smoke explicitly selects an available Secure A40 at **$0.49/hour**, with 48 GB VRAM, 50 GB host RAM and 9 vCPUs. The smoke's 3-hour planning value is **$1.54** including running disk; its 4-hour worker timeout corresponds to approximately **$2.06** at that rate, excluding image-pull/startup time and external services. The $0.60/hour allocation guard and worker deadlines are operational limits, not guaranteed billing caps. The subsequent full A40 run uses the measured forecast below and a freshly verified $0.49/hour quote.

## Measured A40 projection after the smoke

The completed 160-image smoke averaged **14.940 seconds per optimizer step** over 40 steps. Baseline generation over 80 images took 14.4 minutes; post-training generation averaged **10.640 seconds per image** over the same 80 images. Periodic 702 MB checkpoints took approximately 35 seconds to save and publish; the final checkpoints include the best adapter and are larger. These measurements include the current constrained decoder and frozen-vision policy.

For two full epochs, scaling the 828 optimizer steps gives **3.44 hours of training**, or **3.60 hours** when scaling by the full workload's slightly higher token count. The measured baseline plus 410 further generations adds approximately **1.45 hours**. Allowing 0.35 hours for container/model startup, less frequent full-run checkpoints and final export gives **5.24–5.40 hours**. Use a **5–6-hour planning range**; this is an extrapolation from a small sample and not a guaranteed runtime. It explains why a complete run can exceed a training-only comparison on two 3090s: single-GPU training and generated evaluation both contribute materially.

| Two-epoch execution | Runtime basis | Compute/running-disk estimate |
|---|---|---:|
| Runpod Secure, one A40 | Measured throughput extrapolated to 5–6 h; observed $0.514/h including disk | **$2.57–$3.08** |
| Runpod Community, one A40 | Same-throughput assumption, earlier $0.35/h GPU quote plus $0.024/h disk; untested | **$1.87–$2.24** |
| Fireworks serverless LoRA | Earlier 2–5 h allowance; token pricing, not hourly rental | **$8.75** token-budget estimate |

Secure remains the selected execution policy. Community is shown only for the earlier requested cost comparison, with no measured Community capacity or throughput. The Fireworks estimate retains the full 512-token generation allowance, whereas the Runpod timing uses observed response lengths. GCS storage/egress, taxes, retries and stopped volumes are additional. The local reproducible calculation is `reports/runpod-secure-smoke-20261002/edugraph-20261002-qwen38-27b-runpod-smoke-v2/performance-projection.json` (ignored). The later L40S benchmark verified the longest training inputs and live external-checkpoint recovery.

## Measured L40S comparison

The authorized [Secure L40S benchmark](RUNPOD-HARDWARE-BENCHMARK.md) used the same trainer image, checkpoints and matched examples. Eighteen warm optimizer steps averaged **11.404 seconds**, versus **15.023 seconds on A40** (1.317× throughput). Twenty paired predictions averaged **8.007 seconds**, versus **10.996 seconds** (1.373×), with 991 versus 999 output tokens. The allocation cost $1.09/hour plus $0.024/hour running disk. Ada was unavailable and remains unmeasured.

| Two full epochs, including generated evaluation | A40 Secure | L40S Secure |
|---|---:|---:|
| Throughput-based forecast, common 0.35 h overhead | 5.24–5.40 h | 4.01–4.14 h |
| Compute and running disk for that forecast | $2.69–$2.78 | $4.47–$4.61 |
| Scheduling allowance | 5–6 h | 4–5 h |
| Cost for scheduling allowance | $2.57–$3.08 | $4.46–$5.57 |

L40S saves approximately **1.2–1.3 hours for $1.78–$1.84 extra** under the paired forecast. Choose it when that turnaround benefit matters; A40 remains the measured lower-cost option. This is an allocation comparison, including host CPU and I/O differences, and not a pure GPU benchmark. L40S took 14.70 minutes from allocation to model-ready; slow cold starts and checkpoint transfers can exceed the common overhead allowance. The trained-checkpoint generation speedup is extrapolated to base-model evaluation. No full-dataset two-epoch run has been measured.

The benchmark restored all 992 optimizer states at step 40, executed the next real batch to step 41, published its resumable checkpoint and passed the longest-input memory probes. It terminated automatically after publication. The independent calculation and raw outputs are in `reports/runpod-hardware-20261002/edugraph-20261002-qwen38-27b-l40s-benchmark-v1/analysis.json` and its hash-verified report bundle.
