# Full Secure A40 training, 2026-10-02

The user authorized a full A40 training run and overnight monitoring after the A40 learning smoke and L40S timing/recovery benchmark. **Pod `khb2ztncw55pbe` was allocated at 21:27:11 UTC / 23:27 CEST on 2026-10-02.** It is a Secure A40 with 50 GB host RAM and 9 vCPUs, at $0.49/hour GPU plus the $0.024/hour running-disk allowance. The model was ready at **21:34:16 UTC** and baseline predictions are progressing. W&B initialization and the GCS startup marker succeeded. The live audit reported 58,363,904 trainable language-adapter parameters and zero trainable vision/bridge parameters. No full-run quality result exists yet.

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

The runtime forecast is **5–6 hours**, approximately **$2.57–$3.08** including running disk, excluding GCS and taxes. Expected completion is around **04:30–05:30 CEST on October 3**, subject to startup, transfer and generation variability. Eight running hours at the observed rate are about $4.11; the watchdog starts inside the worker, so this is not a strict billing cap for provisioning or an unreachable host.

## Preparation evidence

All 1,968 released images and both metadata files were loaded from locally cached bytes and checked against the immutable release tree's object hashes. Preparation re-rendered the full corpus with the pinned processor, found no duplicate bytes, split overlap or conflicting labels, and confirmed 1,686,068 two-epoch training tokens and 828 steps. The full 512-token allowance plus the longest prompt fits the existing 2,048-token guard; the earlier GPU probe generated normally rather than forcing all 512 tokens.

The configuration commit is recorded separately from the trainer commit. A clean Git diff proved `src`, dependencies, lockfile, prompts and schemas identical to the image's original commit. Prepared runtime versions and prompt/schema/template/processor hashes also match the successful smoke. This reuses the verified image without relabeling its code identity. The existing complete suite passed 277 tests with 96.30% coverage; no production trainer code changed for this run.

The immutable input bundle is `gs://edugraph-classify/runpod/inputs/edugraph-20261002-qwen38-27b-runpod-full-a40-v1/3f03907e5e5d4aeb65817f4620abeb4fe6a16ac1f5944ba6b354051fd7ce5be6.tar` (86,343,680 bytes). The rendered request contains only secret references, selects Secure A40 only, and has no resume fields, public ports or account-wide Runpod key.

Local run files are in `runs/secure-full-a40-20261002/edugraph-20261002-qwen38-27b-runpod-full-a40-v1/` (ignored). The durable launch record is under `gs://edugraph-classify/runpod/attempts/<run-id>/khb2ztncw55pbe/launch.json`. [W&B run](https://wandb.ai/edugraph-io/edugraph-classify/runs/edugraph-20261002-qwen38-27b-runpod-full-a40-v1).

## Overnight monitoring

The current-chat heartbeat `monitor-overnight-a40-training` checks every 15 minutes until 08:00 UTC on October 3, ending sooner after completion or a reported stopped failure. It reports meaningful milestones, completion, failure or required input, while recording ordinary progress locally. It does not launch another Pod, alter training, delete checkpoints or deploy a model.

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

The verifier checks archive SHA-256/size, finite language-only weights and optimizer/RNG state, reproduces baseline/both epochs/final metrics from raw outputs, and saves the verified report under `reports/runpod-full-a40-20261002/<run-id>/` and the GCS attempt prefix. Update this record and the existing draft PR with the result, then disable the heartbeat. Do not promote or deploy the model automatically.
