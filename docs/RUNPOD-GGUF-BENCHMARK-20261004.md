# Secure L40S GGUF inference benchmark, 2026-10-04

**Completed, with no model promotion or GCP inference change.** The selected epoch-2 Runpod adapter was merged with its pinned BF16 base, exported to llama.cpp GGUF, quantized to Q4_K_M, and replayed on the original 64-image validation cohort. The first concurrency-one round scored **19/64 exact-set match (29.69%) and 88.49% micro label F1**. The matching NF4-plus-adapter selection result is **43/64 (67.19%) and 95.26% F1**. The 250-image final assessment from the training run is a separate cohort and is not used as this comparison. The merged BF16 model was hashed but not scored independently, so this run does not identify how much of the loss comes from merging, quantization, the llama.cpp serving path, or their interaction.

## Run contract and artifacts

- Run ID: `edugraph-20261004-qwen38-27b-gguf-benchmark-v3`; Secure NVIDIA L40S Pod `vg8sgfoyvohz9h`, one GPU, no public ports; worker code commit `6071d2698aa3b342255293c15063235ff288fae6`.
- Image: `ghcr.io/christian-bick/edugraph-classify-inference-benchmark@sha256:cc11fd49420681839ce27c36b13c6b32c7ad9de63b085ad512e383b521599ee3`; llama.cpp commit `0faee5004297c3bcfa40b7bf11750127b8c1fd7d` (`b11384`).
- [Example of the executed recipe](../experiments/inference-benchmark-qwen38-27b-gguf-v3.json): Q4_K_M, 512 maximum output tokens, 1,024-token microbatch, temperature zero, two warmups, two 64-image repeats each at concurrency 1, 2, and 4; five-hour worker watchdog.
- Source adapter bundle: `gs://edugraph-classify/runpod/models/edugraph-20261002-qwen38-27b-runpod-full-a40-v1/28aa0526e1b693d7bff66d97b8823ff23bc06082bc3eb3d2a1f9d5cb4020ec49.tar` (257,576,960 bytes). The selected training epoch is 2. Base model `Qwen/Qwen3.8-27B` is pinned to revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Dataset release `v0.30.0-03`, revision `9b509867e898490615be3f59bc2f31fac429389c`, ontology `0.30.0`; original validation image IDs, images, labels, prompt, closed schema, chat template and processor are hash-pinned in the prepared manifest. The input bundle is `gs://edugraph-classify/runpod/inputs/edugraph-20261004-qwen38-27b-gguf-benchmark-v3/f205e231359bf766b1042f04c48b169f3561e0de104dade3e170266690c789da.tar` (25,671,680 bytes).
- Verified report: `gs://edugraph-classify/runpod/benchmarks/edugraph-20261004-qwen38-27b-gguf-benchmark-v3/5fed9f60a7f80cfea48dc65a856efc672447b129eaf649a52875d3cfb8026ad8.tar` (1,361,920 bytes). It contains `result.json` with raw responses, normalized predictions, per-case assessments, round metrics, timings, export hashes and runtime observations. A downloaded, verified copy is under gitignored `reports/gguf-benchmark-v3/`.
- Candidate model bundle: `gs://edugraph-classify/runpod/inference-models/edugraph-20261004-qwen38-27b-gguf-benchmark-v3/62a7ab79647efa18678b85d89457fa6eea3bc3836ba37b56bd4602918501eaf0.tar` (17,478,901,760 bytes). It contains the Q4_K_M language GGUF, BF16 vision projector, prompt/schema/template and reference hashes. The language GGUF SHA-256 is `bd2ec1357e84b27b58ec0b1c83e143028f9b54e42989dc9f3db7c59aed870639`; the projector SHA-256 is `3248a0391bed86eef3842121dab70d4657da244bde47ae02ce37580351bd0138`. This is a benchmark candidate, not a deployed model.

The worker recorded all merged Hugging Face shard hashes and the BF16 language GGUF hash (`492282b16f6247d7e447cebcc9decd92c9de9175298bf13cbb76e7ac0e531eca`). It did not preserve those large intermediates as separate durable model bundles. No merged-BF16 inference score exists.

## Accuracy on the matched validation cohort

The first concurrency-one round is the reference result below. Explicit and canonicalized aggregate metrics were identical. Exact-set match is primary; precision, recall and F1 are micro-averaged over explicit dimension labels. Two raw outputs were invalid because of duplicate labels, which deterministic canonicalization removed. Derived ontology closure was not scored.

| Configuration | Images | Exact-set match | Precision | Recall | F1 | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| Selected epoch-2 NF4 + adapter, training validation | 64 | 43/64 (67.19%) | 94.93% | 95.60% | 95.26% | 1/64 |
| Merged then Q4_K_M GGUF, llama.cpp, concurrency 1 repeat 1 | 64 | 19/64 (29.69%) | 89.69% | 87.32% | 88.49% | 2/64 |

The 24-case exact-set difference is **37.50 percentage points**. A paired case audit found 27 cases correct only under NF4 and three correct only under GGUF; 24 of those 27 NF4-only cases differ in the abilities dimension. This is a diagnostic cluster, not evidence that a particular conversion step caused the regression. The second concurrency-one repeat had the same aggregate quality metrics; concurrency-two repeats scored 18/64 and concurrency-four repeats 19/64, with F1 around 88.5–88.7%. This small concurrency variation is another reason to retain raw predictions and treat one deterministic setting as the comparison target.

| GGUF slice, concurrency 1 repeat 1 | Exact-set match | Label F1 |
|---|---:|---:|
| Areas | 53/64 (82.81%) | 93.69% |
| Scopes | 38/64 (59.38%) | 93.17% |
| Abilities | 29/64 (45.31%) | 61.08% |
| Question views | 7/32 (21.88%) | 88.48% |
| Solution views | 12/32 (37.50%) | 88.51% |

Label-macro F1 was 79.13%. Labels appearing one to four times in training had five true positives, two false positives and two false negatives; this is too small for a stable rare-label estimate. There were no gold labels unseen during training, although the model predicted 13 such labels, so unseen-label recall is unmeasured.

## Runtime, capacity and cost

The Pod reported **46,068 MiB (44.988 GiB)** total GPU memory, **188.0 decimal GB** effective host RAM and **13.6** effective CPUs. An earlier 45 GiB minimum rejected this normal L40S before model work; the executed v3 recipe uses a 44 GiB minimum. Peak sampled GPU use was **18,840 MiB** of 46,068 MiB, with mean sampled GPU utilization of **51.46%** across 752 samples. This establishes fit for the tested L40S configuration, not for the 24 GB GCP L4 or a production concurrency/load pattern.

| Concurrency, repeat 1 | Server startup to healthy | Median per-image latency | p95 per-image latency | Throughput |
|---|---:|---:|---:|---:|
| 1 | 4.68 s | 2.28 s | 2.64 s | 0.445 images/s |
| 2 | 31.26 s | 3.52 s | 4.29 s | 0.571 images/s |
| 4 | 28.85 s | 5.91 s | 7.98 s | 0.681 images/s |

The second repeats were close: median latencies 2.30, 3.59 and 6.03 seconds; throughput 0.448, 0.565 and 0.667 images/s. Per-image latency includes image hashing, JPEG conversion, request construction and server response; it is not pure model execution. Server startup above measures an already allocated Pod starting `llama-server`; it is not a Cloud Run cold start. The complete worker interval recorded in the completion marker was **3,239.6 seconds (54.0 minutes)** at a reviewed **$1.09/hour compute + $0.05/hour running disk**, yielding a **$1.0259 estimate** for that interval. Of this, `result.json` records 2,523.6 seconds and $0.7991 before final publication; the later interval includes artifact upload. Both estimates exclude time before the worker timer, retained GCS storage, tax and other provider charges. Actual billed spend was not independently measured. The Pod terminated automatically after publication; a fresh Runpod list found no active Pod. The two earlier 45 GiB preflight attempts, Pods `20kqn58g7jy54m` and `poh7npbq99xeoi`, were removed after failing before model work; their charges are not included in the v3 worker estimate.

## Interpretation and next check

The pinned llama.cpp JSON-schema path emits properties in their request order ([schema conversion](https://github.com/ggml-org/llama.cpp/blob/0faee5004297c3bcfa40b7bf11750127b8c1fd7d/common/json-schema.cpp), [grammar conversion](https://github.com/ggml-org/llama.cpp/blob/0faee5004297c3bcfa40b7bf11750127b8c1fd7d/common/json-schema-to-grammar.cpp)). All 384 GGUF responses in this run started with `abilities, areas, scopes`; the supervised targets and NF4 outputs used `areas, scopes, abilities`. This is a generation-context mismatch even though the evaluator treats dimensions as sets. The benchmark request builder has since been adjusted to send an ordered shallow copy of the same closed schema with `areas, scopes, abilities`, preserving its meanings and the original file. That correction has **not** been measured on a GPU; the v3 report and hashes still describe the original request. A new controlled run would be needed to learn whether it recovers accuracy. A merged BF16 evaluation would then help separate numerical conversion effects from serving behavior. Neither rerun nor model promotion is authorized by this record.
