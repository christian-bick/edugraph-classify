# Secure A40 learning smoke, 2026-10-02

Status: **completed and Pod terminated**. Held-out label F1 improved from **3.18% to 61.25%**, exceeding a 41.67% image-independent frequency baseline. Exact-set match remained **0/64**, including after canonicalization. This is sufficient evidence that the training path learns on unseen images, but not a model promotion or proof of production quality. Training stopped after the planned single epoch; full training remains unlaunched.

## Reproducible allocation

| Item | Observed value |
|---|---|
| Run | `edugraph-20261002-qwen38-27b-runpod-smoke-v2` |
| Code | `7e51615ca0f6edfd548918dfd2b87c180fd6d258` |
| Public image | `ghcr.io/christian-bick/edugraph-classify-trainer@sha256:d77c8fffdeeb6c0091b7b58a44174b278d48ddac60f9ffa1955d1f2ab1843e64` |
| Input bundle SHA-256 | `584ab63238eef416585bbbfcf109940284aecab6f7ccbb023ccfc6f7cca91610` |
| Pod | `9eukpngd5r2ek8`, verified `secureCloud: true` |
| Hardware | One NVIDIA A40, nominal 48 GB VRAM, 50 GB host RAM, 9 vCPUs |
| Usable device memory | 47.40399169921875 GiB reported by PyTorch |
| Rate | $0.49/hour GPU, $0.514/hour observed total including running disk |
| Allocation start | 2026-10-02 16:48:01 UTC |
| Container start | 2026-10-02 16:50:26 UTC |
| Model ready | 2026-10-02 16:53:14 UTC |
| Final publication | 2026-10-02 17:36:07 UTC; 48.11 minutes after allocation |
| Termination verified | Pod absent and account spend $0/hour at 17:36:43 UTC; rechecked at 17:40:08 UTC |
| Trainable parameters | 58,363,904 language-adapter parameters; zero vision/bridge parameters |

The [recipe](../experiments/edugraph-20261002-qwen38-27b-runpod-smoke-v2.json) pins dataset v0.30.0-03, ontology 0.30.0, Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, compact system prompt v3, NF4 QLoRA and constrained generated evaluation. It uses 160 training images, 64 validation images and 16 training diagnostics, with 40 optimizer steps in one epoch. The final assessment cohort is unused. Only the execution memory threshold changed from 48 to 45 GiB before launch to account for the A40's ECC/driver reservations; GPU selection and training policy were unchanged.

The published image's anonymous pull, Linux/amd64 platform, digest and embedded code commit were verified before launch. [Image build](https://github.com/christian-bick/edugraph-classify/actions/runs/37034760191). Local verification covered 272 tests and 96.30% coverage; one unrelated Windows file-lock error passed on an isolated rerun. The real W&B SDK reference-artifact test does not request Google ADC.

## Learning and output quality

| Held-out metric, 64 images | Frozen NF4 baseline | After 40 steps |
|---|---:|---:|
| Exact-set match | 0/64 | 0/64 |
| Label micro-F1 | 3.178% | 61.248% |
| Precision | 5.200% | 57.385% |
| Recall | 2.289% | 65.669% |
| Raw invalid output rate | 18.75% | 12.50% |

An offline control always predicted the most frequent training labels, using each dimension's rounded mean training cardinality (2 areas, 5 scopes, 1 ability). It used no validation labels to select that prediction and scored **41.667% F1, 0/64 exact matches**. The trained model exceeds it by 19.58 percentage points. This is a useful learning control, not an image-ablation experiment or a confidence interval. The 16 training diagnostics improved from 3.209% to 60.976% F1; their exact-set match also remained zero.

After training, dimension F1 was 48.95% for areas, 66.58% for scopes and 52.94% for abilities. Question-view F1 was 56.71%, versus 65.59% for solution views. Rare labels seen 1–4 times in training reached 43.68% F1; labels unseen in the smoke training cohort reached only 7.69%. Full per-label-frequency and per-view reports are retained in the export. The reserved final-assessment cohort was not used.

Raw baseline outputs used allowed ontology terms but mostly selected incorrect labels. Duplicate labels affected **12/64 baseline outputs and 8/64 trained outputs**; deterministic deduplication repaired their contract validity without changing set-based F1 or exact-set match. No Markdown, malformed JSON or unknown-label failure was observed in either held-out pass. `lm-format-enforcer` does not enforce the schema's `uniqueItems` keyword in this configuration: constrained generation still requires semantic/duplicate validation. The decoding policy was unchanged throughout the paired experiment. Remaining zero exact-set accuracy reflects semantic errors, not label ordering or duplicate handling.

The exported raw predictions were re-evaluated independently on the workstation against the original prepared cohort. Both full held-out metric objects matched the worker's reports exactly.

## Performance and cost

Baseline held-out generation averaged 10.426 seconds per image (median 7.311, p95 39.234); the longest responses repeatedly emitted the same labels. After training, held-out generation averaged 11.207 seconds (median 11.135, p95 15.338). Baseline generation over all 80 images took approximately 14.4 minutes and post-training generation took 14.2 minutes. The GPU runtime reported one bitsandbytes fallback for a matrix dimension not aligned to its fast kernel; its isolated performance effect was not measured.

All 40 optimizer steps averaged **14.940 seconds**, totaling 9.96 minutes of optimizer training. Mean loss fell from 1.971 over steps 1–5 to 0.559 over steps 36–40; final-step loss was 0.391. Peak PyTorch allocation reached **29.122 GiB** on the 47.404-GiB device. A 24 GB GPU cannot run this implementation unchanged. These measurements do not prove that the longest full-dataset example fits, or that a 32 GB card has sufficient driver/allocator headroom.

Runpod's observed account-balance decrease was **$0.4161 for v2** and **$0.5074 including the earlier failed startup**. The observed account rate matched this Pod throughout the measurement. This excludes GCS storage/egress and is account-balance evidence, not an invoice. The Pod and its volume were automatically removed; ongoing Runpod spend is zero. The measured two-epoch A40 projection is **5–6 hours, $2.57–$3.08 Secure including running disk**; see the calculation and comparison in [training costs](TRAINING-COSTS.md).

## Checkpoint verification

The step-zero bundle was 233,963,520 bytes and the step-ten bundle was 701,706,240 bytes including optimizer state. Independent downloads verified their SHA-256 values. The final continuation checkpoint is **935,424,000 bytes**, including the best adapter, optimizer state, RNG state, data position and reports; the model export is **256,921,600 bytes**. Both final bundles were also downloaded and checksum-verified.

The final checkpoint contains 992 finite language-only adapter tensors and 992 matching finite AdamW states, all at optimizer step 40, with epoch 1 complete and offset 0. CPU/CUDA/Python RNG state and resume compatibility match the original manifest. A local continuation check accepted a new run ID with a total epoch target of two, retaining all other training/data/code identities. This verifies external recovery data and compatibility, not a live GPU restart. No additional training was launched.

Seven checkpoint bundles total **4,911,636,480 bytes** in GCS; the model export adds 256,921,600 bytes. W&B holds small reference JSON artifacts, not duplicate tensor files. Checkpoint retention remains explicit pending live recovery validation.

## Tracking and recovery

- [W&B run](https://wandb.ai/edugraph-io/edugraph-classify/runs/edugraph-20261002-qwen38-27b-runpod-smoke-v2).
- Prepared inputs: `gs://edugraph-classify/runpod/inputs/edugraph-20261002-qwen38-27b-runpod-smoke-v2/584ab63238eef416585bbbfcf109940284aecab6f7ccbb023ccfc6f7cca91610.tar`.
- Launch and worker markers: `gs://edugraph-classify/runpod/attempts/edugraph-20261002-qwen38-27b-runpod-smoke-v2/9eukpngd5r2ek8/`.
- Verified result: the same attempt prefix's `verified-result.json`; local copy in `reports/runpod-secure-smoke-20261002/edugraph-20261002-qwen38-27b-runpod-smoke-v2/result-summary.json`.
- Completion marker: `gs://edugraph-classify/runpod/runs/edugraph-20261002-qwen38-27b-runpod-smoke-v2/completed.json`.
- Final model SHA-256: `4dadae8abf5e81225255b5a9ab4fb79485a0293c27adcaada4dbaa50f765b9c9`.
- Continuation SHA-256: `1c21059c6608cf48247e883771af13735ed853f06a4b5e5246a01244b08477d5`.
- Local prepared manifest and sanitized observations: `runs/secure-a40-20261002/edugraph-20261002-qwen38-27b-runpod-smoke-v2/` (ignored).
- Local allocation record: `reports/runpod-secure-smoke-20261002/edugraph-20261002-qwen38-27b-runpod-smoke-v2/launch.json` (ignored).

The worker saves resumable checkpoints every ten steps and each epoch, with optimizer/RNG/data position and the current adapter kept together. Success publishes the selected model to GCS, flushes W&B and terminates the Pod. Failure stops it, retaining the billable recovery volume. The four-hour watchdog begins at worker startup; it does not cover image pulling. Inspect the actual Pod state after completion.

## First startup attempt

The earlier v1 Pod `yhbqhq55hz4qkx`, code `a69514d`, failed before model loading because its automatically supplied Pod-scoped key receives 403 on REST v1. The workstation stopped it and saved the failure record, then verified GraphQL read/stop/terminate with that same scoped key and removed the empty Pod volume. No optimizer steps ran. Account balance decreased by approximately $0.09 during that startup; Runpod reported zero ongoing spend before the v2 launch.

The worker now uses GraphQL for its scoped lifecycle calls while the workstation retains REST v1 with its account key. No account-wide Runpod key is sent into a Pod. Full details and continuation commands are in [Runpod training](RUNPOD-TRAINING.md).
