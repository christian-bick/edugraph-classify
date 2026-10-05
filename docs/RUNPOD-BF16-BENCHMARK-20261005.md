# Secure A100 merged-BF16 inference benchmark, 2026-10-05

**The corrected v2 run completed on the original 64-image validation cohort: 40/64 exact-set match (62.50%) and 95.23% micro label F1 in each of two repeats, with no invalid outputs.** The selected epoch-2 NF4-plus-adapter model scored 43/64 (67.19%) and the corrected Q4_K_M GGUF v4 run scored 37/64 (57.81%) on the same IDs and gold labels. The first BF16 attempt (v1) had failed on its first warmup, before producing a score. Neither BF16 run promoted a model or changed GCP inference.

## Completed v2 run and artifact pins

- Run ID `edugraph-20261005-qwen38-27b-bf16-benchmark-v2`; Secure NVIDIA A100-SXM4-80GB Pod `4d0yt6amr6xtej`, one GPU, no public ports; code commit `43fb369a737e7a4a725e00ad13a4c48691643020`; image `ghcr.io/christian-bick/edugraph-classify-inference-benchmark@sha256:8b01e636ab99b396c86577bfcb840acd5486db576acf1a2af1f8f1cdd48310cf`.
- [Executed v2 recipe](../experiments/inference-benchmark-qwen38-27b-bf16-v2.json): `merged_bf16_hf`, 512 maximum output tokens, greedy constrained generation, two warmups, two 64-image repeats at concurrency one, and a five-hour worker watchdog. The request added `TORCH_DISABLE_NATIVE_JIT=1`, as used by the Runpod training request; it retained the source model and benchmark settings of v1.
- Selected epoch-2 adapter bundle `gs://edugraph-classify/runpod/models/edugraph-20261002-qwen38-27b-runpod-full-a40-v1/28aa0526e1b693d7bff66d97b8823ff23bc06082bc3eb3d2a1f9d5cb4020ec49.tar` (257,576,960 bytes). Base `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; dataset `v0.30.0-03` revision `9b509867e898490615be3f59bc2f31fac429389c`; ontology `0.30.0`. The prepared manifest pins the original validation IDs, images, labels, prompt, closed schema, chat template and processor.
- Input bundle `gs://edugraph-classify/runpod/inputs/edugraph-20261005-qwen38-27b-bf16-benchmark-v2/d1be083396a807dbacded20d134ecefb59ab75364618bf0dd5c94e02701e22bf.tar` (25,671,680 bytes). The report and independent paired replay checked the original 64-case cohort and contract against the pinned NF4 and GGUF v4 reports.
- Downloaded and SHA-256-verified report `gs://edugraph-classify/runpod/benchmarks/edugraph-20261005-qwen38-27b-bf16-benchmark-v2/7b67069179af27ac38d87bc1ab95de8062f54cd42b70b67182da4d27fe702c38.tar` (368,640 bytes; SHA-256 `7b67069179af27ac38d87bc1ab95de8062f54cd42b70b67182da4d27fe702c38`). The unpacked `result.json` SHA-256 is `1faa6aec175f25b23d488050e0f3353988c5242bef2ef19d5a39ffd8efbd7c79`. The report, completion marker and paired replay are under gitignored `reports/bf16-benchmark-v2/`; no merged checkpoint was published as a candidate artifact.

The 22 merged Hugging Face file hashes yield digest `d89447e12cc44abf94fe9ae51bc1e62d78dd5e28d78e451764ea9d19f03703d6`, identical to v1 and the corrected GGUF v4 export. This confirms that the BF16 and GGUF measurements started from the same merged checkpoint contents.

## Matched validation results

The first BF16 repeat is the reference below; repeat two had the same raw generated text and token IDs, per-case results and aggregate quality metrics. All 128 BF16 responses stopped normally. Explicit and canonicalized aggregate metrics matched. Exact-set match is primary; precision, recall and F1 are micro-averaged over explicit dimension labels, without scoring ontology-derived closure.

| Configuration | Exact-set match | Precision | Recall | F1 | Invalid |
|---|---:|---:|---:|---:|---:|
| Selected epoch-2 NF4 + adapter | 43/64 (67.19%) | 94.93% | 95.60% | 95.26% | 1/64 |
| Merged BF16, Transformers, A100 | 40/64 (62.50%) | 95.57% | 94.89% | 95.23% | 0/64 |
| Q4_K_M GGUF v4, llama.cpp, L40S | 37/64 (57.81%) | 94.61% | 92.78% | 93.69% | 0/64 |

Against NF4, BF16 shared **38** correct cases, recovered **two** NF4 errors, regressed on **five** NF4-correct cases, and shared **19** wrong cases: a net three-case (4.69-point) drop. Against corrected GGUF v4, BF16 shared **35** correct cases, recovered **five**, regressed on **two**, and shared **22** wrong cases: a net three-case (4.69-point) gain. The paired counts and metrics reproduced from the saved raw predictions in both BF16 repeats.

| Exact-set slice | NF4 | BF16 | GGUF v4 |
|---|---:|---:|---:|
| Abilities | 58/64 | 58/64 | 56/64 |
| Areas | 54/64 | 57/64 | 54/64 |
| Scopes | 48/64 | 48/64 | 43/64 |
| Question views | 19/32 | 17/32 | 16/32 |
| Solution views | 24/32 | 23/32 | 21/32 |

## A100 runtime and interpretation

The Pod reported 81,920 MiB (80 GiB) total GPU memory, 125.0 decimal GB effective host RAM and 27.2 effective CPUs. Peak sampled GPU use was 53,977 MiB, with 40.86% mean sampled utilization across 700 samples. Merging took 1,406.2 seconds; full-GPU model loading took 73.9 seconds. At concurrency one, the first repeat took 346.4 seconds (median/p95 per-image latency 5.54/7.12 seconds; 0.185 images/s), and the second took 330.0 seconds (5.35/6.65 seconds; 0.194 images/s). BF16 per-image latency covers processor batch creation, generation and decoding; image rendering happens before the measured rounds. The A100 timings are not directly comparable with llama.cpp on an L40S and do not establish Cloud Run cold-start or fit on the current 24 GB GCP L4.

The completion marker records a **2,338.4-second (39.0-minute) worker interval** and a **$1.0653 compute-plus-running-disk estimate** at $1.59/hour compute plus $0.05/hour disk. It excludes image provisioning before the worker timer, retained GCS storage, tax and other provider charges; actual billed spend was not independently measured. The Pod terminated after publication: a subsequent Runpod account list had zero Pods and a GET by this Pod ID returned 404.

BF16's three-case deficit to NF4 cannot be attributed to merging alone because the base-weight precision also changed. BF16's three-case lead over GGUF v4 cannot be attributed to quantization alone: conversion, llama.cpp serving and JSON constraint mechanics also differ. A BF16 GGUF replay through the same serving path would isolate quantization more closely. The 64-case cohort and differing A100/L40S runtimes make this a diagnostic result, not a promotion or production-capacity test.

## Failed v1 attempt and verified artifacts

- Run ID `edugraph-20261005-qwen38-27b-bf16-benchmark-v1`; one Secure NVIDIA A100-SXM4-80GB Pod `l2zy019l222zbu`, no public ports, code commit `ca427d40e82254c01e792bb659a7a46f78f586eb`, image `ghcr.io/christian-bick/edugraph-classify-inference-benchmark@sha256:59190e509f2435f2411314c8cfe33c87dddf6e4387e1024b23dd89ecbcccc5df`.
- [Executed example recipe](../experiments/inference-benchmark-qwen38-27b-bf16-v1.json): `merged_bf16_hf`, original 64-image validation cohort, 512-token output ceiling, two warmups, two repeats, concurrency one, greedy constrained generation, and a five-hour worker watchdog. The original selected NF4-plus-adapter validation score is 43/64 exact; the corrected v4 Q4_K_M GGUF score is 37/64 on these same IDs. Neither is a BF16 result.
- Source adapter bundle `gs://edugraph-classify/runpod/models/edugraph-20261002-qwen38-27b-runpod-full-a40-v1/28aa0526e1b693d7bff66d97b8823ff23bc06082bc3eb3d2a1f9d5cb4020ec49.tar` (257,576,960 bytes). Base `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; dataset `v0.30.0-03` revision `9b509867e898490615be3f59bc2f31fac429389c`; ontology `0.30.0`. The prepared manifest pins the original images, labels, prompt, schema, processor and their hashes.
- Input bundle `gs://edugraph-classify/runpod/inputs/edugraph-20261005-qwen38-27b-bf16-benchmark-v1/28b65866cbf2bd291d494321b1ab63a6d4d278861e043f384b4e8c6acb8ce2d3.tar` (25,671,680 bytes). Its 78 non-recipe hashed inputs matched the corrected GGUF v4 run.
- Verified failure bundle `gs://edugraph-classify/runpod/attempts/edugraph-20261005-qwen38-27b-bf16-benchmark-v1/l2zy019l222zbu/02270d4fc723efcd6609c71f0842d001d8e84cda3e8b2d6cfb0c14ab4b174888.tar` (10,240 bytes; SHA-256 `02270d4fc723efcd6609c71f0842d001d8e84cda3e8b2d6cfb0c14ab4b174888`). The downloaded and verified copy, failure marker, sanitized traceback and progress record are under gitignored `reports/bf16-benchmark-v1/`.

## V1 failure and resolution

The progress record reached `scoring` after a 1,259.5-second merge. Its merged file-hash digest was `d89447e12cc44abf94fe9ae51bc1e62d78dd5e28d78e451764ea9d19f03703d6`, exactly the digest pinned from the corrected GGUF export. `settings` was empty: the first warmup failed before any measured round or prediction. The sanitized `RuntimeError` traceback enters Qwen rotary-position computation, PyTorch's native Triton `bmm` outer-product implementation, and Triton's `compile_module_from_src` / `_build` path. The exception message was deliberately suppressed, so a missing C compiler is a plausible cause, not a confirmed one. The v1 evidence does not implicate checkpoint merging and supplied no BF16-to-GGUF quality comparison.

The training Runpod request already sets `TORCH_DISABLE_NATIVE_JIT=1`. The separately approved v2 request added that flag while preserving the model, cohort, renderer and generation settings, and completed both repeats. This supports the native-JIT path as the cause of the v1 failure, although the sanitized exception message does not prove that a C compiler was missing.

The Pod exited after failure and was explicitly terminated. A subsequent Runpod account list showed no remaining Pod, so no active Pod or disk rental remains from this run. Reviewed rates were $1.59/hour compute and approximately $0.05/hour running disk; actual billed spend was not independently measured.
