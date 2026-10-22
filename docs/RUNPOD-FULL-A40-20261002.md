# Full Secure A40 training, 2026-10-02

**Completed and independently verified on 2026-10-03.** Two epochs over all 1,654 training images finished with 828 optimizer updates. Epoch 2 was selected on the 64-image selection cohort; the separate 250-image final assessment achieved **72.4% exact-set match (181/250)** and **96.44% label F1**, with zero invalid outputs. The model and continuation bundles are verified in GCS. The Pod terminated automatically and fresh Runpod API checks confirmed zero ongoing account spend. No model promotion or inference deployment occurred.

The user authorized this run and overnight monitoring after the A40 learning smoke and L40S timing/recovery benchmark. Pod `khb2ztncw55pbe` used a Secure A40, 50 GB host RAM and 9 vCPUs, at $0.49/hour GPU plus the $0.024/hour running-disk allowance. Allocation began at **21:27:11 UTC / 23:27 CEST on October 2**; the completion marker was published at **02:46:01 UTC / 04:46 CEST on October 3**. Allocation to publication took **318.84 minutes (5 hours 19 minutes)**. The observed Runpod account-balance decrease was **$2.73155**, excluding GCS and tax; this is not an invoice.

## Verified learning results

All baseline, epoch and final metric dictionaries were reproduced locally from the exported raw predictions and original prepared examples. Exact-set match is the primary outcome; label precision, recall and F1 are micro-averaged. Label ordering does not affect the scores.

| Phase and cohort | Images | Exact-set match | Label precision | Label recall | Label F1 | Invalid outputs |
|---|---:|---:|---:|---:|---:|---:|
| Frozen NF4 baseline, selection | 64 | 0/64 (0%) | 5.20% | 2.29% | 3.18% | 12/64 |
| Epoch 1, selection | 64 | 33/64 (51.56%) | 91.84% | 95.07% | 93.43% | 2/64 |
| Epoch 2, selection | 64 | 43/64 (67.19%) | 94.93% | 95.60% | 95.26% | 1/64 |
| Selected epoch 2, final assessment | 250 | 181/250 (72.40%) | 96.66% | 96.22% | 96.44% | 0/250 |

The final assessment was evaluated only after selecting epoch 2. Its score is a separate cohort result, not an additional improvement measured on the 64 selection images. Final explicit and canonicalized scores are identical: 2,115 true-positive, 73 false-positive and 83 missed labels. No ontology closure was added. Earlier invalid outputs contained duplicate labels; deterministic deduplication repaired validity without changing those phases' aggregate exact-set or F1 scores. The final cohort required no such repair.

| Final assessment slice | Exact-set match | Label F1 |
|---|---:|---:|
| Areas | 93.60% | 97.21% |
| Scopes | 80.40% | 96.59% |
| Abilities | 92.00% | 94.80% |
| Question views, 125 images | 74.40% | 96.75% |
| Solution views, 125 images | 70.40% | 96.14% |

Label-macro F1 is **90.19%**, below micro F1. Labels seen 1–4 times in training achieved 92.59% F1, but that slice contains only 27 gold occurrences (25 correct, two missed, two extra). The final cohort contains no gold labels unseen during training, so it provides no evidence of unseen-label recall. The separate 16-image training diagnostic fell from 75% exact-set / 95.08% F1 after epoch 1 to 62.5% / 90.24% after epoch 2; selection remained governed by held-out predictions. These results establish learning on the released assessment cohort, not universal accuracy or automatic production readiness. Sixty-nine final images still have at least one incorrect or missing label.

Subsequent offline error analysis on October 3 informed the [refinement plan](REFINEMENT-PLAN.md), including numerical-specificity ambiguities, local error clusters and limited combination coverage. The final cohort was excluded from this run's checkpoint selection, so the historical result stands. Its reuse for methods motivated by that analysis is exploratory or regression testing; selected follow-up improvements require fresh final assessment.

## Runtime, memory and durable artifacts

All 828 optimizer-step events were observed without gaps. Their reported durations total **223.98 minutes**, averaging **16.23 seconds per update**. The 490 generated predictions total **79.05 minutes** of recorded latency: 13.32 minutes baseline, 13.48 after epoch 1, 12.73 after epoch 2 and 39.51 for final assessment. The remaining **15.81 minutes** include provisioning/loading, checkpointing, export and other overhead. Final assessment latency was **9.63 seconds median / 13.14 seconds p95**. The training-only average is about 112 minutes per epoch; the complete run includes substantial evaluation work.

Peak PyTorch allocation was **29.949 GiB**. Both the runtime audit and downloaded adapter names confirmed **58,363,904 trainable language-adapter parameters**, **992 adapter tensors**, and **zero trainable vision/bridge parameters**. This does not establish that a 32 GB device fits the whole process; retain the validated nominal 48 GB configuration.

GCS contains **20 checkpoint bundles**: baseline step 0, every 50 steps through 800, epoch boundaries at 414 and 828, and a separate final-assessment continuation at 828. The trained periodic snapshots arrived about every 13–15 minutes, with longer intervals during epoch evaluation. No stall, failure or timeout intervention was needed.

| Verified artifact | Bytes | SHA-256 |
|---|---:|---|
| Selected model bundle | 257,576,960 | `28aa0526e1b693d7bff66d97b8823ff23bc06082bc3eb3d2a1f9d5cb4020ec49` |
| Continuation bundle | 935,782,400 | `4664fbf8c65615e40e0c3212fd14a287e57a409adaad7ff165bacb28e9974444` |

The model is at `gs://edugraph-classify/runpod/models/<run-id>/<model-sha256>.tar`; continuation is at `gs://edugraph-classify/runpod/checkpoints/<run-id>/<continuation-sha256>.tar`. The model includes the selected adapter, processor, compact prompt, closed schema, embedded chat template, manifest and all raw prediction/metric reports. The independent verifier downloaded both archives, checked their sizes and SHA-256, checked all adapter tensors and AdamW moments for finite values, and confirmed all 992 optimizer states at step 828 plus RNG state. Continuation position is epoch 2, offset 0, phase `final`, with epoch 2 selected. This bundle passed local recovery compatibility checks; no GPU resume of this full-run bundle was executed. The earlier L40S benchmark separately demonstrated live GPU recovery.

The complete verified report is `gs://edugraph-classify/runpod/attempts/<run-id>/khb2ztncw55pbe/verified-result.json`, with local copies and downloaded archives under `reports/runpod-full-a40-20261002/<run-id>/` (ignored). The completion marker is `gs://edugraph-classify/runpod/runs/<run-id>/completed.json`. Read-only REST and GraphQL checks independently found the Pod absent; GraphQL reported `currentSpendPerHr: 0` and an unchanged final balance on a subsequent check. Retained GCS artifacts still incur their normal storage charges.

## Run contract

- Run ID: `edugraph-20261002-qwen38-27b-runpod-full-a40-v1`.
- [Committed recipe](../experiments/edugraph-20261002-qwen38-27b-runpod-full-a40-v1.json), configuration commit `f82e79026c3bbb7043c7244a705b4d3dc1249f2a`.
- Unchanged trainer commit `7e51615ca0f6edfd548918dfd2b87c180fd6d258` and image `ghcr.io/christian-bick/edugraph-classify-trainer@sha256:d77c8fffdeeb6c0091b7b58a44174b278d48ddac60f9ffa1955d1f2ab1843e64`.
- Dataset v0.30.0-03, revision `9b509867e898490615be3f59bc2f31fac429389c`; ontology 0.30.0; Qwen/Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Fresh initialization from the pinned base, with NF4 double quantization and BF16 compute. The smoke adapters are not the starting weights.
- Compact v3 system prompt, rank 8 / alpha 16, language-only attention/MLP adapters, frozen vision and bridge, microbatch one / effective batch four, constant learning rate 1e-4. These training settings match the tested implementation.
- All 1,654 training images, two epochs, 828 optimizer updates. Selection uses 64 validation images; 16 training diagnostics are also generated per evaluation. The separate 250-image assessment cohort is used only after selecting the best epoch.
- Baseline plus both epoch evaluations and final assessment: 490 generated classifications, with the existing JSON constraints and 512-token output ceiling. Selection compares exact-set match, then F1; loss alone cannot select or promote a model.
- GCS snapshots every 50 steps, at epoch boundaries and before final assessment. Each includes optimizer, RNG, training position, current adapter, reports and the selected adapter when available.
- Eight-hour worker watchdog; $0.60/hour GPU allocation guard. Completion publishes external artifacts, flushes W&B and terminates the Pod. Failure stops it while retaining the billable volume for recovery.

The prelaunch forecast was **5–6 hours / $2.57–$3.08** including running disk, excluding GCS and taxes. The measured 5.31 hours / $2.73 fell inside that range. Eight running hours at the observed rate would have been about $4.11; the watchdog starts inside the worker, so this was not a strict billing cap for provisioning or an unreachable host.

## Preparation evidence

All 1,968 released images and both metadata files were loaded from locally cached bytes and checked against the immutable release tree's object hashes. Preparation re-rendered the full corpus with the pinned processor, found no duplicate bytes, split overlap or conflicting labels, and confirmed 1,686,068 two-epoch training tokens and 828 steps. The full 512-token allowance plus the longest prompt fits the existing 2,048-token guard; the earlier GPU probe generated normally rather than forcing all 512 tokens.

The configuration commit is recorded separately from the trainer commit. A clean Git diff proved `src`, dependencies, lockfile, prompts and schemas identical to the image's original commit. Prepared runtime versions and prompt/schema/template/processor hashes also match the successful smoke. This reuses the verified image without relabeling its code identity. The existing complete suite passed 277 tests with 96.30% coverage; no production trainer code changed for this run.

The immutable input bundle is `gs://edugraph-classify/runpod/inputs/edugraph-20261002-qwen38-27b-runpod-full-a40-v1/3f03907e5e5d4aeb65817f4620abeb4fe6a16ac1f5944ba6b354051fd7ce5be6.tar` (86,343,680 bytes). The rendered request contains only secret references, selects Secure A40 only, and has no resume fields, public ports or account-wide Runpod key.

Local run files are in `runs/secure-full-a40-20261002/edugraph-20261002-qwen38-27b-runpod-full-a40-v1/` (ignored). The durable launch record is under `gs://edugraph-classify/runpod/attempts/<run-id>/khb2ztncw55pbe/launch.json`. [W&B run](https://wandb.ai/edugraph-io/edugraph-classify/runs/edugraph-20261002-qwen38-27b-runpod-full-a40-v1).

## Overnight monitoring

The current-chat heartbeat `monitor-overnight-a40-training` checked every 15 minutes overnight, reporting meaningful milestones and recording ordinary progress locally. Monitoring ended after completion, independent verification and publication of the results. No further Pod was launched, training changed, checkpoint deleted or model deployed.

Local scheduled checks require the computer to stay on and the desktop app to keep running. The remote worker independently handles checkpointing, its timeout and successful-publication termination. [Official scheduling requirements](https://learn.chatgpt.com/docs/automations?surface=app).

From the repository root, collect sanitized container logs and a read-only health snapshot:

```powershell
$run = 'runs/secure-full-a40-20261002/edugraph-20261002-qwen38-27b-runpod-full-a40-v1'
uv run --no-sync python temp/runpod-secure/observe.py "$run/pod-launch.json" --tail
uv run --no-sync python temp/runpod-full/monitor.py
```

Use `--all --tail` on the log observer for provisioning/system logs. `health-latest.json` records status, observed training/evaluation progress, external checkpoint metadata and account spend. CUDA memory-utilization percentage is bandwidth utilization, not allocated VRAM. During loading, checkpoint saves and generation, short idle periods are normal. Confirm apparent stalls with fresh logs and runtime; do not stop a healthy run merely because there is no new checkpoint during an evaluation. A genuine 30-minute progress stall, a failed worker or the exceeded deadline warrants stopping this Pod and preserving its recovery state.

After durable completion, verify automatic termination and zero ongoing spend. If shutdown failed after successful publication, verify the external model and continuation bundles before terminating the completed Pod. For failure, stop compute and retain the volume until recovery evidence is secure; do not remove the volume merely to clear an alert.

Independent post-run verification uses the existing artifact/evaluation functions:

```powershell
uv run --no-sync python temp/runpod-full/retrieve.py $run checkpoint
uv run --no-sync python temp/runpod-full/retrieve.py $run model
uv run --no-sync python temp/runpod-full/monitor.py
uv run --no-sync python temp/runpod-full/verify_result.py
```

The verifier checks archive SHA-256/size, finite language-only weights and optimizer/RNG state, reproduces baseline/both epochs/final metrics from raw outputs, and saves the verified report under `reports/runpod-full-a40-20261002/<run-id>/` and the GCS attempt prefix. These checks passed. The report includes `quality_promotion: false`; deployment remains a separate decision.
