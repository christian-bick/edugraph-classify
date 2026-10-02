# Local Docker training assessment

This document retains the historical 9B proposal. The current single-GPU 27B QLoRA executor, W&B tracking and recovery workflow are implemented separately in [Runpod training](RUNPOD-TRAINING.md), pending GPU validation. For memory requirements and the cost comparison, see [training costs](TRAINING-COSTS.md). Shared conversion takes the tracked prompt verbatim, and the provider-neutral helper builds the separate ontology-pinned decoding schema. The historical two-GPU launch path below remains a proposal.

Registry update (2026-10-02): new self-hosted images use **GHCR**, with GCS retained for run artifacts. The historical training repositories were deleted in the authorized [GCP cleanup](GCP-CLEANUP.md); references below describe the original proposal and are no longer pullable. Use the new [container publishing workflow](CONTAINER-IMAGES.md) and record its image digest in future run configurations. Public GHCR images need no pull credentials; private images need GHCR access independently of the ADC used for GCS. Production inference remains on GCP with scale-to-zero.

Assessed 2026-09-13 for a standalone box with two RTX 3090s. User-confirmed boundary: training starts directly on both GPUs; GCP remains the provider for surrounding services, including artifact storage and model upload after training. The single-GPU milestone is removed. This is a proposed implementation scope; no local provider, host setup, upload, or GPU training run was created by this assessment.

## Recommendation

Implement a `local_docker` training adapter with execution mode `self_hosted` and target **`Qwen/Qwen3.5-9B` on both RTX 3090s from the first training diagnostic**. Use one coordinated two-process DDP job with NF4 QLoRA, bfloat16 compute, rank 8, gradient checkpointing, and microbatch size 1 per GPU. Reuse the existing trainer and GCS inputs/outputs, adapting the training path for distributed execution. GCP supplies the pinned container and receives the completed model artifacts. The [historical 4B Vertex failure](VERTEX-AI-TRAINING.md#operational-smoke-history-and-runtime-diagnosis) informs diagnostics but does not require a separate 4B or single-GPU milestone.

## Selected model and initial recipe

The user selected Qwen3.5-9B after clarification of the model name. The official [Qwen catalog](https://huggingface.co/api/models?author=Qwen&search=Qwen3.8&limit=100), checked on 2026-09-13, contained no 8–10B Qwen3.8 checkpoint; its smallest published Qwen3.8 checkpoint was 27B. The selected [Qwen3.5-9B model](https://huggingface.co/Qwen/Qwen3.5-9B) is a vision-language model with 9,653,104,368 total parameters according to its Hub metadata, which explains the approximate 10B display.

| Setting | Initial planned value |
|---|---|
| Model repository | `Qwen/Qwen3.5-9B` |
| Immutable model revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Architecture | `Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5` |
| Training executor | `local_docker`, execution mode `self_hosted` |
| Hardware / world size | 2 × RTX 3090, one node, two processes |
| Initial distributed backend | DDP with one quantized model replica per GPU |
| Method / precision | Supervised QLoRA, NF4 with double quantization, bfloat16 compute and quantized storage |
| LoRA | Rank 8, alpha 16, dropout 0.05, `all-linear` targets |
| Learning rate | `2e-5` |
| Train / validation microbatch | 1 per GPU |
| Gradient accumulation | 2 steps per process; effective training batch size 4 |
| Gradient checkpointing | Enabled, `use_reentrant=False`; model cache disabled |
| Targets / thinking | Assistant label JSON only; thinking disabled in the training template |
| Initial duration / seed | One-batch distributed diagnostic, then one short epoch; seed 42 |
| Artifacts | Persistent local working directory, completed outputs published to the GCS run prefix |

The revision and architecture were verified through the [model API](https://huggingface.co/api/models/Qwen/Qwen3.5-9B) and [pinned configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json). Start implementation against the repository's locked training dependencies and verify model-class resolution, multimodal processing, quantization, and distributed kernels in the actual image. Sharing the `qwen3_5` architecture with the old 4B checkpoint does not establish runtime compatibility or memory fit.

The new [`edugraph-20260913-qwen35-9b-local-ddp-v1.json`](../experiments/edugraph-20260913-qwen35-9b-local-ddp-v1.json) recipe pins these identities with dataset v0.26.0-01, its source ontology v0.26.0, and the semantically identical v0.27.0 ontology client. It supports provider-neutral preparation of the full official splits; it is not yet a launchable local job. The [dataset audit](DATASET-UPGRADE-0.26.0-01.md) confirms eligibility and image integrity and records the remaining within-training gold-set inconsistency. Historical 4B configurations and revisions remain intact. Fix image-pixel and expanded-token limits from a representative data/processor profile before launching, and record those limits and any recipe change in a new configuration.

## GCP and local responsibilities

| Responsibility | Proposed location |
|---|---|
| Immutable prepared inputs, manifests, and published run records | Existing GCS bucket in `europe-west4` |
| Pinned training container | Existing GCP Artifact Registry repository |
| Model loading, training steps, and in-training validation loss | One coordinated Docker training job using both RTX 3090s, with one process per GPU |
| Active checkpoints, model cache, and logs during training | Persistent disk on the box, with GCS publication as configured |
| Completed model/adapter, processor, and reproducibility metadata | Upload to the run's GCS output prefix after training |
| Subsequent quality evaluation, model registration, and serving | GCP remains the default; these workflows still need their own implementation |

The intended flow is GCS inputs and an Artifact Registry image → local training → GCS model/artifact upload → downstream GCP workflows. Local disk is working and recovery storage; an entirely local artifact backend is no longer required for the first milestone. The public upstream model and dataset identities remain pinned regardless of where their copies are stored.

Record training execution and artifact services separately. `local_docker` / `self_hosted` identifies the training executor, while the GCP project, bucket, and image registry identify its supporting services. The preparation recipe carries the latter under `provider.artifact_services` and hardware under `provider.hardware`; the manifest preserves both as configuration data. Strict local launch validation and shared staging remain to be implemented. Do not label a local run as `gcp_vertex_ai` / `serverless` merely to pass the existing staging validator.

## Hardware fit

Each [RTX 3090 has 24 GB of memory](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/). DDP gives each card its own quantized 9B model replica. A bounded 9B QLoRA recipe is a plausible initial design, but fit and throughput remain unmeasured. Image resolution, expanded vision tokens, unquantized modules, optimizer state, and activations make a parameter-count estimate insufficient. Measure peak memory on both ranks within the first two-GPU diagnostic, including representative and longest records.

| Execution choice | What it provides | Recommendation |
|---|---|---|
| Two GPUs with DDP | One model replica per card and synchronized gradients | Initial implementation and first model diagnostic |
| Two GPUs with FSDP-QLoRA | Shards parameters and training state, with additional dtype, wrapping, and checkpoint integration | Planned memory fallback if the measured 9B workload cannot fit per card |

DDP does not turn the two cards into a single 48 GB allocation. If a required record exceeds the per-card budget, record the failure and move the same 9B/two-GPU target to a separately pinned FSDP-QLoRA configuration. FSDP-QLoRA is supported in the Hugging Face stack, but generic text-model examples do not prove that this exact multimodal architecture and locked versions work. Verify dtype alignment, PEFT wrapping of trainable/frozen modules, loading peaks, and checkpoint save/reload. The trainer already sets `bnb_4bit_quant_storage` to bfloat16, which is only one prerequisite. See [bitsandbytes FSDP-QLoRA](https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora) and the [locked PEFT version's FSDP guidance](https://huggingface.co/docs/peft/v0.20.0/en/accelerate/fsdp).

The initial scope is supervised QLoRA of the selected 9B model. Keep model identity, data, and explicit-label targets fixed while diagnosing the two-GPU backend; do not silently substitute the historical 4B model, switch to independent GPU jobs, or fall back to a single GPU.

## Host prerequisites and unknowns

Assume a Linux x86-64 host, preferably Ubuntu 24.04, running Docker Engine. If the box runs Windows/WSL2, validate GPU passthrough, NCCL, filesystem behavior, and recovery separately before treating it as equivalent. The box's OS, driver, RAM, storage, CPU, GPU topology, and access method have not been inspected.

- Install an NVIDIA driver supporting the cards and the existing CUDA 12.6 container, plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Docker support alone does not provide GPU access. Configure the Docker NVIDIA runtime and verify both devices inside the exact training image.
- NVIDIA lists a CUDA 12.x minor-compatibility floor of driver branch 525, with feature and PTX caveats. Prefer a maintained driver that supports CUDA 12.6 without relying on those exceptions; verify kernels and bitsandbytes in the container. The host needs the driver and container runtime, while the CUDA user-space libraries stay in the image. [Compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
- Plan for at least 64 GB host RAM and roughly 200 GB free SSD/NVMe space as initial engineering allowances, with 128 GB RAM useful for larger datasets or CPU offload. These are planning assumptions, not measured minimums. Model caches, container layers, decoded images, checkpoints, and retained runs determine the actual requirement. The current loader materializes decoded train and validation images in RAM; two processes duplicate this.
- Inspect `nvidia-smi`, `nvidia-smi topo -m`, free memory, PCIe links, and any installed NVLink bridge. Test NCCL collectives under the selected container. NVLink is optional for the proposed DDP milestone and does not remove the need for explicit sharding when memory is the constraint.
- Check sustained cooling, PSU capacity, and power availability for both cards. Record wall time and measured energy separately from managed-provider charges; local compute still consumes electricity.
- Use persistent host directories for input bundles, model cache, checkpoints, and logs, with bounded container shared memory and appropriate ownership. Cache the pinned public model revision so repeat diagnostics do not download it again. Record image digest, lock hash, driver/CUDA versions, GPU identity, seed, world size, precision, and effective batch size in each run.

For the first version, run the CLI on the box over an existing SSH session. Remote management from the Windows workstation can follow through a named [Docker SSH context](https://docs.docker.com/engine/security/protect-access/), with credentials held by SSH. The trainer can read staged manifests and JSONL directly from GCS. Any cached inputs or checkpoint bind mounts must refer to paths on the Docker host; a remote context does not automatically transfer Windows files.

The box needs two authentication paths: Docker authentication to pull the private Artifact Registry image, and ADC available to the process that reads/writes GCS. Host login alone does not make credentials available inside the container. Configure an appropriate keyless identity for the on-premises/container environment and inject its credential configuration at runtime; keep credentials out of images, manifests, and logs. Grant access to the existing training-image repository and intended storage locations, without requiring Vertex CustomJob creation privileges for this local executor. See [Google's ADC setup guidance](https://docs.cloud.google.com/docs/authentication/provide-credentials-adc) and [Artifact Registry Docker authentication](https://docs.cloud.google.com/artifact-registry/docs/docker/authentication).

## Repository gaps and proposed changes

| Boundary | Current implementation | Required local support |
|---|---|---|
| Provider identity | `ExecutionMode.SELF_HOSTED` and generic model/run identities already exist | Identify local training independently of its GCP storage, registry, and downstream services |
| Preparation | `prepare_run` supports the aligned v0.26.0-01 dataset through explicit `labels` selection; the new recipe records the 9B/two-GPU executor and GCP service settings | Reuse the prepared bundle in shared staging; choose and record runtime image-pixel/token limits before launch |
| Artifact transport | `build_vertex_staging_plan` couples reusable GCS staging to a strict `gcp_vertex_ai` / `serverless` provider check | Extract shared GCS staging/runtime-manifest construction; preserve Vertex-specific validation at the Vertex job boundary |
| Runtime output | `RuntimeConfig.from_mapping` accepts GCS output; `run` already uploads the saved model and results | Retain GCS output, provide ADC, persist the working directory, and make publication retryable without retraining |
| Container | `containers/vertex-trainer/Dockerfile` is already a Linux/amd64 CUDA image using the locked Python stack | Pull the pinned image from Artifact Registry; share the image recipe and training loop |
| Job lifecycle | Managed adapters implement cloud discovery and launch records | Add strict local job config, offline rendering, preflight, start/status/logs/stop, stable container/run identity, collision checks, timeout, and recovery records |
| Provider protocols | `TrainingProvider` uses Fireworks-shaped upload/SFT records; `ServerlessTrainingProvider` is explicitly managed | Introduce a small container-job lifecycle protocol only where shared operations warrant it; do not make local training emulate Fireworks uploads or call it serverless |
| Device placement | `_train_qlora` loads the quantized model with `device_map={"": 0}`; CUDA preflight inspects device 0 | Select the process-local CUDA device before model loading, validate both GPUs, and require world size 2 for this plan |
| Distributed execution | No `torchrun` entrypoint or explicit rank-safe artifact handling | Launch one process per GPU, select `LOCAL_RANK`, verify Trainer/Accelerate data sharding, synchronize failures, and guard processor/result publication to rank zero |
| Checkpoint recovery | Trainer saves once per epoch; no exposed resume policy; local work defaults to `/tmp` | Persist step checkpoints with optimizer/scheduler/RNG state, record their hashes, and support explicit resume; GCS backup/publication can supplement local recovery |
| Observability/evaluation | Secret-safe stages and one-batch diagnosis exist; training returns loss/log history | Preserve these, add per-rank memory/timing and durable failure records, publish artifacts to GCP, and keep later quality evaluation on the GCP path |

The learning loop and GCS I/O can remain shared between Vertex and local execution. The main reuse obstacle is the staging planner's assumption that GCS implies a Vertex serverless job. Extract that common service boundary while keeping provider-specific job validation explicit. An alternative local storage backend is optional future work.

The existing `_upload_output` uses create-only writes for each output file. A partial upload followed by a retry currently collides with objects already written. Reuse the hash/size-aware retry pattern in `GcsArtifactStore`, preserve completed files on the box, and allow publication to resume independently of training. Record training completion separately from artifact publication; upload the adapter/model, processor, base-model revision, manifests, and result metadata, then write a completion record only after all expected objects are verified. For QLoRA, identify the output as an adapter unless a merged model was explicitly produced. Upload to GCS is artifact publication; any later Vertex model registration remains a separate GCP workflow.

For the initial DDP job, use `torchrun --standalone --nnodes=1 --nproc-per-node=2` or an equivalent pinned Accelerate configuration, with both selected devices visible to the container. Override the image's current Python entrypoint appropriately. Set the CUDA device from `LOCAL_RANK` before loading quantized weights, and use that same device for diagnostic transfers. Validate DDP unused-parameter handling with the actual LoRA target set; multimodal modules may not all participate in every batch. Do not use inference-oriented `device_map="auto"` as a distributed-training implementation. Validate the locked PyTorch/Trainer/PEFT behavior with a real two-rank run. [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html).

The planned effective batch size is `1 microbatch × 2 accumulation × 2 processes = 4`. This matches the historical batch size while the model and execution configuration change. Record both per-device and global settings explicitly. Any full-state checkpoint operation required collectively by a distributed backend must still involve its required ranks; rank-zero publication is not a blanket instruction to skip all save operations elsewhere.

The existing one-batch diagnostic bypasses `Trainer.train()` and manually performs an optimizer step. Adapt it to the selected distributed backend before the first GPU run. Both ranks must participate in the same collective-aware forward/backward/optimizer step, with rank-specific stages, finite-loss/gradient checks, and a bounded collective timeout. Verify synchronized trainable parameters for DDP, or valid shard ownership and reconstructable adapter state for FSDP. Two independently executed diagnostics do not establish coordinated training. Diagnose the unresolved model failure in this two-GPU path; completing the old 4B single-GPU smoke is not a prerequisite.

## Implementation and acceptance sequence

1. **Shared GCP services and two-GPU executor:** separate GCS staging from Vertex job validation; implement local Docker configuration, two-rank launch, rank-aware loading, preflight, persistent working directories, and lifecycle handling together. Include sampler behavior, coordinated failure/timeout, and exactly-once result publication. Unit-test with fake processes and GCS clients, including provider identities, hashes, collisions, and partial-upload retries; retain at least the configured 95% coverage.
2. **Container, topology, and GCP checks:** build from `uv.lock`, verify imports and the chosen model processor, and validate fixtures offline. Verify both GPUs and NCCL collectives in the actual container. Separately verify authenticated image pull and GCS input reads, then test output publication/retry with a small authorized artifact. The selected configuration must fail clearly if either requested GPU or a required rank is unavailable.
3. **First optimizer step on both GPUs:** after training is authorized, run the adapted distributed one-batch diagnostic on the selected model. Verify forward, finite loss, backward, collective communication, and optimizer updates on both ranks. Measure peak allocated/reserved memory on representative and longest records. Record image-pixel and token budgets explicitly; preserve legibility and reject over-limit data rather than silently dropping tokens. Resolve runtime errors within this two-GPU configuration.
4. **Short two-GPU training and recovery:** use a separate technical fixture for plumbing or an aligned released dataset for quality work. Verify both devices contribute, sample distribution is intentional, effective batch size is controlled, and adapter/checkpoint save and reload work. Exercise process failure, checkpoint resume, and recovery of coordinated state before long runs. Technical smoke data establishes no quality baseline.
5. **GCP model handoff:** verify complete publication of the adapter, processor, pinned base-model identity, manifests, and results. Exercise interrupted upload/retry without repeating training. Verify canonical JSON generation and explicit-label validity in the downstream GCP evaluation workflow.
6. **Quality evaluation:** report exact-set match, precision/recall/F1, per-dimension, question/solution, frequency, invalid-output, latency, and cost/energy diagnostics. Any memory-driven backend or precision change receives a new recorded configuration and another two-GPU diagnostic.

The first milestone includes distributed training: a two-GPU executor, shared GCP staging, host authentication, coordinated diagnostics and recovery, and reliable model upload. The aligned dataset is now available and validated. The unresolved model runtime failure remains to be diagnosed directly on the chosen model and both GPUs. Evaluation tooling also remains incomplete, so successful training establishes the execution path before it establishes model quality.
