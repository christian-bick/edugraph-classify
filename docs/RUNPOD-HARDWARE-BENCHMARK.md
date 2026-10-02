# Secure GPU timing and recovery benchmark

Status: **prepared and staged; no benchmark Pod launched**. The 18:29 UTC capacity check on 2026-10-02 returned no matching Secure RTX 6000 Ada allocation. The user's hardware preference is pending: retain the prepared Ada run or use an available Secure L40S. All 277 local tests passed with 96.30% production-code coverage. Current Runpod spend remains zero.

The user authorized a bounded RTX 6000 Ada benchmark on 2026-10-02 after the A40 smoke. This does not launch full training. The target is one Secure RTX 6000 Ada 48 GB, with at least 48 GB host RAM / 8 vCPUs, the existing 170 GB disk allowance, a $1/hour allocation guard and a one-hour worker watchdog. At the current $0.84/hour GPU price, a half-hour planning estimate is $0.432 including running disk. Startup precedes the worker watchdog and these are operational limits, not guaranteed billing caps. Capacity must be checked immediately before launch.

The benchmark reuses the exact A40 image digest and trainer commit `7e51615ca0f6edfd548918dfd2b87c180fd6d258`. A separately committed and SHA-256-verified [experiment harness](../experiments/benchmarks/runpod_hardware.py) replaces only the orchestration entrypoint. It calls the existing worker with a diagnostic executor, retaining its Secure allocation, price, bundle, image-code and lockfile checks, GCS/W&B integration, watchdog, failure stop and publication-before-termination behavior. The harness is fetched from its immutable public Git commit and verified before execution. The prepared bundle separately records that harness commit/hash and the diagnostic specification hash. No account-wide Runpod credential is injected.

The [local preparation script](../experiments/benchmarks/prepare_runpod_hardware.py) verifies the original smoke artifacts, refuses changed trainer sources/dependencies, and reuses their original cohort and runtime identity. Additional memory-probe images are independently hash-verified from the full dataset audit. They are separate from the resumed training inputs so checkpoint compatibility is unchanged.

## Measurements

1. Restore the external step-zero A40 checkpoint and replay exactly its first 20 optimizer batches. Compare steps 3–20 with the matching A40 steps; the first two are warm-up on both sides. Preserve losses and batch identifiers as well as timing.
2. Restore the external epoch-one/step-40 checkpoint, including optimizer and RNG state. Generate two warm-ups followed by 20 fixed held-out classifications already present in the A40 export. Compare paired latency, output token counts and exact text agreement. Hardware-dependent output differences remain visible.
3. Restore step 40 again to remove sampling's RNG effects, process the actual next epoch-two batch, check all optimizer states advance to 41, and publish a coherent continuation checkpoint at epoch 1 / offset 4 / step 41. This is an external-checkpoint restore onto a new GPU Pod. It is not a complete second epoch.
4. Run one discarded training update on the longest question and solution training examples, repeated to fill the effective batch of four; generate normally from the longest prompt using the original trained checkpoint. Record allocated, reserved and device-used VRAM. Normal generation uses the 512-token ceiling but does not force an answer to consume all 512 tokens. No final-assessment example is used.
5. Publish the report and flush W&B, then terminate the Pod. The memory-probe update is excluded from the saved continuation. The diagnostic never promotes a model or publishes a new selected model.

The two-epoch recipe ceiling exists solely to validate continuation from epoch one; the harness always executes the bounded steps above. It records 22 optimizer updates across independent branches, not 22 sequential continuation steps. Timing and recovery branches are separate. W&B uses an increasing diagnostic event index, with the true optimizer position in each event's fields.

The benchmark tests use fake providers and a tiny checkpoint engine to verify bounded execution, exact batch order, step-40 restoration, step-41 publication, exclusion of memory-only updates, finite optimizer checks, and failure behavior. The original trainer and production inference configuration remain unchanged.

## Prepared Ada run

- Run ID: `edugraph-20261002-qwen38-27b-ada-benchmark-v1`.
- Harness commit: `c2050ec3676feec1ee52759438373209807065b8`; source SHA-256 `ad78bf082118a542a87965a324949bc9c72fc4c55cbb1c90b84630be832686a7`.
- Input bundle: `gs://edugraph-classify/runpod/inputs/edugraph-20261002-qwen38-27b-ada-benchmark-v1/6475ca8c85bc6e14eb01508c7a5b4c9f374960935e43ceb840fd37f032d831e3.tar` (30,423,040 bytes).
- The 18 matched A40 training steps average 15.022954 seconds. The 20 matched A40 predictions average 10.995955 seconds and contain 999 output tokens in total.
- Memory probes have 1,281 and 1,302 training tokens; the largest prompt plus the full generation allowance is 1,772 tokens.
- Local manifest, specification, rendered request and capacity observations: `runs/secure-ada-20261002/edugraph-20261002-qwen38-27b-ada-benchmark-v1/` (ignored).

No Ada throughput, live recovery or longest-example GPU result exists yet. The preparation verifies the public harness download/hash, original trainer sources, external checkpoint compatibility, image bytes and rendered inputs locally.
