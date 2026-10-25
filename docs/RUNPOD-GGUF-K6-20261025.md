# K–6 Qwen3.8-27B GGUF benchmark — 2026-10-25

**Completed and locally verified.** The candidate and report archives, all required members, training controls, repeated predictions, paired comparison and provider shutdown evidence passed verification.

Run `edugraph-20261025-qwen38-27b-k6-gguf-v2` completed conversion and matched validation on a Secure L40S. Both repeats achieved **43/64 exact-set match (67.19%)**, **95.38% label F1**, and **zero invalid outputs**. The selected NF4 source achieved 47/64 on these same images. This is a reproducible export result with a documented quality loss; it does not establish production readiness. Hugging Face publication has not been attempted for this candidate, and Imagine/GCP production remains unchanged.

## Source, scope and execution

The [K–6 training run](RUNPOD-K6-A40-20261024.md), `edugraph-20261024-qwen38-27b-runpod-k6-v1`, selected epoch 2 after two epochs, 1,166 optimizer steps and 2,330 training images. It trained language-only rank-8/alpha-16 adapters, with vision and the multimodal bridge frozen.

The training corpus covers **in-scope Common Core State Standards for Mathematics from Kindergarten through Grade 6 (K–6)**. The [Coverage Explorer](https://coverage.edugraph.io) documents competency coverage and scope exclusions. This describes corpus scope, not verified mastery of every standard.

| Source identity | Pin |
|---|---|
| Dataset | `christian-bick/edugraph-exercises`, `v0.31.0-01`, revision `00ae2441669a42d1357c70d6f194bc71286eb464` |
| Ontology | `0.31.0`; semantic snapshot `14034fa5790c4e5977ce8da72ee4dd34213b1d06b6d15f71ee539bc42361be6e` |
| Base model and processor | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Training code | `1a9e34c131fa9ebcd56fee70d5c6c75c0e913f26` |
| Selected source archive | SHA-256 `57e381ec0a7026e70775beca47e9732554fc50562533a78bb7b8f33cffe04e6a`; 257,986,560 bytes |

The retry preserved the [v1 export configuration](../experiments/inference-benchmark-qwen38-27b-k6-gguf-v1.json), changing only the run identity. It merged the selected adapter into the pinned BF16 base, converted with llama.cpp `0faee5004297c3bcfa40b7bf11750127b8c1fd7d`, quantized language weights to **Q4_K_M**, and retained the **BF16 vision projector**. The [first attempt](RUNPOD-GGUF-K6-STARTUP-20261025.md) had no observed worker startup and remains a separate stopped allocation.

The benchmark used the same 64 checkpoint-selection images, gold labels, image hashes and training-frequency reference as NF4: two repeats at concurrency one, following two warmup images. Configuration was 4,096-token context, logical batch and microbatch; 512 output tokens; greedy generation; RGB JPEG quality 75; and request schema order `areas`, `scopes`, `abilities`. The larger microbatch accommodates the selection cohort's 1,248 maximum image tokens; the full prepared corpus reaches 2,625. These are configuration and preparation checks, not independent measurement of every server-side rendering.

## Quality and paired errors

| Configuration | Exact-set match | Precision | Recall | Micro F1 | Invalid |
|---|---:|---:|---:|---:|---:|
| Selected NF4, epoch 2 | 47/64 (73.44%) | 97.33% | 96.23% | 96.78% | 0/64 |
| GGUF repeat 1 | 43/64 (67.19%) | 95.47% | 95.29% | 95.38% | 0/64 |
| GGUF repeat 2 | 43/64 (67.19%) | 95.47% | 95.29% | 95.38% | 0/64 |

All 64 generated response texts are byte-identical between repeats. Each repeat has 506 true-positive, 24 false-positive and 25 false-negative labels. Explicit and canonicalized metrics agree; one response per repeat only required sorting its abilities labels. All 128 responses use the expected property order and finish with `stop`, with no truncation or invalid labels. Example-mean F1 is 94.45%; label-macro F1 is 88.71%, versus NF4's 96.90% and 90.72%. Macro F1 covers labels observed in gold or predictions, not the full ontology; no ontology closure is credited.

| Dimension/view | GGUF exact | Precision | Recall | F1 | NF4 exact / F1 |
|---|---:|---:|---:|---:|---:|
| Areas | 59/64 | 97.74% | 96.30% | 97.01% | 58/64 / 97.01% |
| Scopes | 50/64 | 94.25% | 95.47% | 94.86% | 54/64 / 96.92% |
| Abilities | 60/64 | 96.43% | 93.10% | 94.74% | 60/64 / 95.91% |
| Question views | 23/32 | 96.50% | 95.75% | 96.12% | 24/32 / 96.48% |
| Solution views | 20/32 | 94.51% | 94.85% | 94.68% | 23/32 / 97.06% |

For labels occurring at least five times in training, GGUF precision/recall/F1 are 95.56%/95.74%/95.65% (495/23/22 TP/FP/FN; 517 gold occurrences), versus NF4 F1 96.99%. The one-to-four occurrence slice has 91.67%/78.57%/84.62% (11/1/3; only 14 gold occurrences), versus NF4 F1 88.89%. There are no unseen gold or predicted labels, so unseen-label recall is unmeasured. Frequency slices restrict label sets across the same images; they are not separate image cohorts.

Each paired repeat has **40 both-correct, 14 both-wrong, seven regressions and three recoveries**. The net loss is four cases: **6.25 exact-match percentage points and 1.40 micro-F1 points**. Five regressions affect scopes, one abilities and one areas; five are solution views and two question views.

Regressions include missing number-range labels on three-digit comparison, `PhysicalNumbers` on equal groups, and `NumbersWithoutZero` on make-ten addition; extra `ArabicNumerals`/`Base10` on conversion problems; and four extra scopes plus missing `DirectRelation` on measured-length differences. Data-attribute description replaces `InformationExtraction`/`TextualArticulation` with `Interpretation`; composite volume loses `Equation`/`Multiplication`. Recoveries occur on arithmetic-pairs fluency, sorting by count, and place-value subtraction explanation. These are gold-relative comparisons, not independent visual adjudications of annotation correctness. Full case identities remain in `paired-validation.json`; the source prediction reports retain the labels used to derive these differences.

The end-to-end difference cannot be attributed solely to quantization: merge, conversion, serving and constrained-decoding mechanics also differ. No matching standalone BF16 diagnostic was run. The historical GGUF's 37/64 is from another release/cohort and does not establish improvement here.

The separate **NF4** final assessment remains 306/414 exact (73.91%), precision 96.62%, recall 95.77%, F1 96.19%, invalid zero. **GGUF has not been evaluated on those 414 images.** Official validation omits 147 of 403 training labels; 224 images across selection and final together reuse bytes from the previously inspected assessment. These are regression measurements, not fresh blind confirmation. Realized-range annotation, label completeness and fresh assessment were deferred.

## Runtime, resources and cost

Pod `lnknvcgufopv9q` ran on machine `xqs1i118vxad` in `EUR-IS-2`, with one Secure L40S, provider allocation 125 GB RAM/16 vCPUs, 30 GB container disk and 350 GB volume. Worker measurements were 124,999,999,488 RAM bytes, 13.6 effective CPUs and 44.988 GiB GPU capacity.

The provider start anchor was **16:03:51.500 UTC**; exact creation time was not retained. The GCS startup marker was created at **16:07:35.072**, completion at **16:52:24.486**. Successful termination was confirmed by HTTP 404 at **16:52:25.887509**, and independently at **16:52:49.666646**. There was no failure marker or watchdog stop request. The watchdog recorded terminal absence and its process was no longer present; exact OS exit time was not retained.

The approved three-hour **allocation** window began at 16:03:50.294688, with a 20-minute unconfirmed-startup cutoff and fixed 18:58:50.294688 stop cutoff before the 19:03:50.294688 absolute limit. Completion occurred well within this window; the worker's later-starting timer did not extend it.

| Measurement | Repeat 1 | Repeat 2 |
|---|---:|---:|
| Measured inference duration | 140.577 s | 139.766 s |
| Throughput | 0.455 images/s | 0.458 images/s |
| Median / p95 latency | 2.148 / 2.569 s | 2.182 / 2.704 s |
| Prompt / completion tokens | 32,947 / 2,634 | 32,947 / 2,634 |

Server startup took 6.613 seconds; quantization reported 524.918 seconds. Per-image latency includes local image preparation and the HTTP/server response. Serving samples peaked at **18,896 MiB (18.453 GiB)** of 46,068 MiB, averaging 62.62% GPU utilization over 258 samples. Sampling starts after export, so this is not a merge/conversion memory peak; temperature was not recorded. The converter's `Unknown RoPE type: default` warning was reviewed against its pinned implementation and preserved section/theta metadata; no execution failure followed.

Worker execution through artifact publication took **2,689.097 seconds**, including 529.166 seconds beyond the report's pre-publication timer. Its $0.85155 estimate excludes initial image/input startup. From the provider start anchor through completion, elapsed time was **48m32.986s**, estimating **$0.88199 compute plus $0.04046 running disk = $0.92245**. Through first confirmed absence the estimate is $0.92289; using the earlier prelaunch anchor gives $0.92327. Rates are $1.09/hour compute plus a $0.05/hour disk allowance, not an invoice; GCS and taxes are excluded. The separate failed v1 cost approximately $0.71743 and retains its 350 GB volume, with ongoing storage described in its failure record.

## Artifacts and publication boundary

Execution commit is `7c9db04e3bc3cf9b018bf8912bac82ddc680c29e`; lock SHA-256 is `f140c157c177d9b7f16a3319c80ae30447b2117ef9b3fe6aa3a5be840851389c`. The [matching image build](https://github.com/christian-bick/edugraph-classify/actions/runs/40191673855) published `ghcr.io/christian-bick/edugraph-classify-inference-benchmark@sha256:2570eea4b136be1727888e17097bfacf828b42c32cf361b7a451d3a4f9cb46be`.

| Artifact | SHA-256 / size |
|---|---|
| Prepared manifest | `6c91792653ae7915d49fb764d478f9f41aa43d6b0149cf5fe687f8ad35868001` |
| Input archive | `8a271b5f5651eea8fd8d4b51c87729b82f2fa7de7cee345d698cf1b25a240543`; 26,490,880 bytes |
| Locally verified report archive | `51b3a6f5f2d6cca09fc1961d9f9422c52934c27b0d21466a787bb6199294a21e`; 460,800 bytes |
| Locally verified candidate archive | `516192b0cbe8e92f0cbd489dfb7d1869e6872291723f17ea5bdcb392c1e9d1de`; 17,479,106,560 bytes |
| Locally verified language GGUF | `22506fbed9ad33421d7322719eb21dbe9c457aa3cdb816e8a48b44295e2723ff`; 16,547,400,064 bytes |
| Locally verified BF16 projector | `3248a0391bed86eef3842121dab70d4657da244bde47ae02ce37580351bd0138`; 931,145,760 bytes |

The durable report is `gs://edugraph-classify/runpod/benchmarks/edugraph-20261025-qwen38-27b-k6-gguf-v2/51b3a6f5f2d6cca09fc1961d9f9422c52934c27b0d21466a787bb6199294a21e.tar`. It contains `result.json` and `candidate-reference.json`; local bytes match the archive. The supported paired comparator reproduced the saved comparison exactly. The report preserves raw responses, metrics, contract hashes and the 22-file merged-model inventory, including 12 weight shards. Local source artifacts are under `reports/<run-id>/`. The committed [verification receipt](../experiments/runpod-k6-gguf-validation-20261025.json) preserves member hashes and sizes, metrics, paired cases, merged-model inventory, and the sanitized timing/cost derivation. All seven candidate archive members were independently checked against their hashes and identities; the inference contract matches the selected training export.

This completed benchmark record precedes the release recipe, which will pin its immutable commit. The eight public files and pinned Apache-2.0 license must pass staging review before the authorized publication to `christian-bick/EduGraph-Classifier-Qwen3.8-27B-GGUF` guarded by predecessor `477e3caf32eced2a9e002ead1edccc9a5c458f0c`. Record any later Hub revision and anonymous file verification in a separate receipt. Benchmark completion and Hub publication do not promote the model to production.

The retained upstream license is [Qwen3.8-27B LICENSE](../experiments/licenses/Qwen3.8-27B-1d4bf0f-LICENSE), SHA-256 `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`, including the Alibaba Cloud notice.
