# Vertex AI serverless training

## Scope

The GCP adapter submits custom-container training through Vertex AI CustomJob. Vertex owns provisioning, monitoring, and teardown, so this repository records the execution mode as `serverless`. A job still specifies ephemeral machine, GPU, disk, replica, timeout, and scheduling values; “serverless” does not mean that compute choices or GPU quota disappear.

This boundary is model-agnostic. It can run a pinned Qwen, Gemma, or DeepSeek VLM trainer without teaching the provider adapter about a particular model framework. Dataset conversion, ontology validation, metrics, and model selection remain outside the adapter.

## Regional and identity policy

Use `europe-west4` for the first setup unless an explicitly recorded experiment decision selects another supported EU region. Keep the following resources in the same region:

- Vertex AI CustomJob;
- Cloud Storage input, checkpoint, and output prefixes;
- Artifact Registry training image;
- optional Vertex TensorBoard and experiment resources.

This is an EU data-plane choice, not a claim that Google is an EU-owned provider.

Use Application Default Credentials (ADC) to submit jobs. For local development, authenticate with `gcloud auth application-default login`; in automated environments, prefer Workload Identity Federation. `GOOGLE_APPLICATION_CREDENTIALS` may point to an approved local credential file, but credential contents must never enter `.env`, a job configuration, a container argument, a launch record, or Git.

Use a dedicated runtime service account. The submitting identity needs permission to create/list Vertex custom jobs and to act as that runtime service account. Grant the runtime service account only the storage, model-download, logging, and artifact permissions required by the trainer. Cross-project Artifact Registry images may require an explicit reader grant to the applicable Vertex service identity.

The project must have the Vertex AI API enabled. The chosen region also needs sufficient custom-training GPU quota. Artifact Registry and Cloud Storage must be enabled when those services host the image and run artifacts.

### Selected GCP setup

The initial GCP project is `edugraph-438718` and the staging bucket is `gs://edugraph-classify`. The bucket location is user-confirmed as `europe-west4`, matching the selected Vertex region. Immutable smoke inputs and runtime manifests are staged under content-addressed prefixes, and job output belongs under `gs://edugraph-classify/runs/<job-id>`.

The dedicated keyless runtime identity is `vertex-training@edugraph-438718.iam.gserviceaccount.com`, and the same-region standard Docker repository is `europe-west4-docker.pkg.dev/edugraph-438718/training`. Both were created with explicit approval. The runtime identity has `roles/storage.objectAdmin` only on `gs://edugraph-classify`, allowing the trainer to read inputs and update checkpoints and outputs without project-wide storage access. The submitting user has `roles/iam.serviceAccountUser` on this service account and effective `iam.serviceAccounts.actAs` permission.

Vertex AI, Artifact Registry, and Cloud Storage APIs are enabled. Host Application Default Credentials are available, and the authenticated principal has the required project permissions to create/get/list CustomJobs, consume service quota, and upload Artifact Registry content, plus create/get/list object access on the staging bucket. The bucket uses uniform bucket-level access with public-access prevention enforced. The bootstrap created no service-account key; credentials remain with host ADC and the managed Vertex runtime identity. Smoke inputs and trainer images have been uploaded and CustomJobs have been attempted, but no trained model has been produced.

## Resolved job configuration

Generated job configurations belong under `runs/<job-id>/` and remain gitignored. They contain reproducibility metadata and infrastructure identifiers, but no secrets. The strict JSON shape is:

```json
{
  "provider_id": "gcp_vertex_ai",
  "training_execution_mode": "serverless",
  "job_id": "edugraph-vertex-smoke",
  "code_commit": "0123456789abcdef0123456789abcdef01234567",
  "project_id": "edugraph-438718",
  "location": "europe-west4",
  "display_name": "EduGraph Vertex smoke",
  "output_uri": "gs://edugraph-classify/runs/edugraph-vertex-smoke",
  "service_account": "vertex-training@edugraph-438718.iam.gserviceaccount.com",
  "container": {
    "image_uri": "europe-west4-docker.pkg.dev/edugraph-438718/training/vlm@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "command": ["python", "-m", "edugraph_trainer"],
    "args": ["--run-manifest", "gs://edugraph-classify/runs/edugraph-vertex-smoke/manifest.json"],
    "environment": {
      "HF_HOME": "/tmp/huggingface"
    }
  },
  "compute": {
    "machine_type": "g2-standard-12",
    "accelerator_type": "NVIDIA_L4",
    "accelerator_count": 1,
    "replica_count": 1,
    "boot_disk_type": "pd-ssd",
    "boot_disk_size_gb": 500
  },
  "scheduling": {
    "strategy": "FLEX_START",
    "timeout_seconds": 14400,
    "max_wait_seconds": 7200
  },
  "labels": {
    "environment": "smoke",
    "workload": "vlm-training"
  }
}
```

The image must be pinned by digest, not a mutable tag. The adapter reserves the `edugraph_job_id` label for collision and retry protection. Secret-like environment names are rejected; the trainer should obtain protected values through its runtime identity and an approved secret manager.

`FLEX_START` requires a positive `max_wait_seconds`. `STANDARD` and `SPOT` require it to be `null`. All strategies require a bounded positive `timeout_seconds`. The example L4 shape is illustrative, not an accepted model-specific training decision; validate regional availability, memory, quota, and expected cost before authorizing a run.

## Offline validation and launch

Render the exact low-level JobService request locally:

```bash
uv run edugraph-classify vertex render \
  --config runs/edugraph-vertex-smoke/vertex-job.json
```

This command needs no GCP credential and performs no external operation.

Launching is a billable external mutation and therefore requires the exact stable job ID:

```bash
uv run edugraph-classify vertex launch \
  --config runs/edugraph-vertex-smoke/vertex-job.json \
  --confirm-job-id edugraph-vertex-smoke
```

The launcher verifies the pinned clean commit, checks for an existing job with the protected label, writes `runs/<job-id>/vertex-launch-record.json` before the create call, and records the returned CustomJob resource name. A retry resumes one unambiguous existing resource after a pending create; it refuses an unrelated collision or multiple matches.

## First Qwen3.5 smoke recipe

The first model is `Qwen/Qwen3.5-4B` at immutable Hugging Face revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The public repository was created on 2026-02-27, is ungated, and declares a 4B causal language model with a vision encoder. Qwen3.5 is natively multimodal and uses thinking mode by default. This meets the requested 2026, VLM, 3B–12B, and reasoning-capable criteria; no Hugging Face credential is needed to download it.

The tracked run is `experiments/edugraph-20260823-qwen35-4b-vertex-smoke-v1.json`. It selects eight examples per official split solely to validate the technical path. Its first-fit recipe is:

- NF4 4-bit QLoRA with bfloat16 compute and gradient checkpointing;
- one epoch, rank 8, alpha 16, dropout 0.05, and learning rate `2e-5`;
- train/evaluation batch size 1 and four-step gradient accumulation;
- all linear modules as LoRA targets;
- prompt/image loss masking with only the canonical assistant JSON supervised;
- thinking disabled in the training chat template because the gold data contains final labels, not reasoning traces;
- W&B disabled; outputs remain in the regional run prefix.

The replacement container uses the official NVIDIA CUDA `12.6.3` cuDNN runtime for Ubuntu 24.04, pinned to its Linux/amd64 manifest digest. Python packages resolve exclusively from the committed `uv.lock`. `torch==2.13.0+cu126` and `torchvision==0.28.0+cu126` come from PyTorch's explicit CUDA 12.6 index; the remaining training pins are `transformers==5.15.1`, `peft==0.20.0`, `accelerate==1.14.0`, and `bitsandbytes==0.50.1`. This deliberately avoids PyPI's default CUDA 13 build and its newer host-driver requirement.

The original compute hypothesis was `g2-standard-12` with one 24 GB NVIDIA L4. Two corrected Flex Start requests exhausted their two-hour capacity wait without provisioning. A subsequent `a2-highgpu-1g` request with one 40 GB A100 provisioned in approximately three and a half minutes, establishing that this shape was available at that time. A100 remains the next smoke-test shape, not a permanent model-specific default. Provisioning wait and execution remain bounded to two hours; OOM must be recorded and resolved explicitly rather than by silent truncation.

## Operational smoke history and runtime diagnosis

The direct Linux/amd64 trainer image digest was accepted by Vertex after an earlier multi-platform image-index issue was removed. Corrected L4 jobs then failed only because no L4 was provisioned within their two-hour Flex Start windows. The A100 job `edugraph-20260825-qwen35-4b-vertex-a100-smoke-v1` (`customJobs/5975489523914637312`) reached `RUNNING`, but its worker exited with status 1 and Vertex retried it four times. The old process boundary recorded only `trainer failed (RuntimeError)`, so the exact failing library call cannot be recovered retrospectively.

The diagnostic replacement job `edugraph-q35-4b-a100-smoke-v2` (`customJobs/6981903303143587840`) provisioned an A100 in under six minutes. CUDA preflight, processor loading, quantized model loading, k-bit preparation, LoRA attachment, and trainer initialization all completed. Each worker attempt then failed deterministically on the first batch. Reproduction with the exact published image and pinned processor showed that the complete record contained expanded multimodal image tokens while its label-masking prefix was encoded with the text-only tokenizer. The two sequences diverged at the image placeholder and triggered the trainer's prefix invariant. The correction processes both the complete record and prompt prefix with the same images and multimodal processor, validates right-padded prefix identity, and derives each mask boundary from the prompt attention mask.

The corrected job `edugraph-q35-4b-a100-smoke-v3` (`customJobs/6720131574802677760`) pinned commit `8219f6d40cc31027869c296bb919b45a045c71eb` and the single-manifest image digest `sha256:cb8a6dc98d039cd68020f9c44f732a31c8f145b9880188ef99a6b2a9159922e1`. It provisioned an A100 in approximately three and a half minutes and cleared the multimodal prefix invariant. The first training step ran for about four seconds, emitted the expected transition disabling model cache for gradient checkpointing, and then failed with a distinct `RuntimeError`. No checkpoint or model object reached the regional output prefix. The secret-safe process boundary correctly withheld the third-party exception message, so another paid retry is blocked until the failure can be reproduced on a GPU or classified through additional bounded, non-sensitive diagnostics.

The remediation is both diagnostic and preventive:

- run a CUDA preflight before processor or model download and report only the pinned PyTorch version, compiled CUDA version, device count, device name, compute capability, and memory;
- emit constant secret-safe stage names for dependency import, CUDA preflight, processor/model loading, quantized-model preparation, LoRA attachment, trainer initialization, training, model/processor save, and output upload;
- preserve only the exception class for unexpected third-party failures, never its potentially sensitive message;
- use the pinned NVIDIA CUDA 12.6/cuDNN base and PyTorch CUDA 12.6 wheels rather than a generic Python base with the default CUDA 13 wheel.
- construct label-mask prefixes with the same multimodal processor and images as the complete records, because tokenizer-only prefixes do not include expanded vision tokens.

### Bounded one-batch diagnostic

The trainer exposes an opt-in `--diagnostic-one-batch` mode for the next A100 investigation. It retains the pinned model, quantization, LoRA, gradient-checkpointing, prompt construction, and optimizer choice, but replaces the opaque full `Trainer.train()` call with exactly one manually staged optimizer step. The stages are collation, device transfer, training-mode selection, fused AdamW initialization, gradient reset, forward, scalar finite-loss validation, backward, and optimizer step. A successful diagnostic writes and uploads only its small run result; it does not evaluate, save a checkpoint, or save a model.

Every diagnostic stage emits a constant start/completion event. A failed stage records only the exception class and at most twelve traceback code locations from the allowlisted `edugraph_classify`, Transformers, PEFT, Accelerate, bitsandbytes, and PyTorch packages. Each location contains only package name, module basename, function name, and line number. Exception messages, source lines, filesystem paths, tensor values, local variables, images, prompts, labels, environment values, and credentials remain suppressed.

Configure this mode offline by adding the diagnostic flag:

```bash
uv run edugraph-classify vertex configure \
  --manifest runs/<job-id>/manifest.json \
  --staging-record runs/<job-id>/vertex-staging-record.json \
  --image-uri europe-west4-docker.pkg.dev/edugraph-438718/training/vlm@sha256:<digest> \
  --output runs/<job-id>/vertex-job.json \
  --diagnostic-one-batch
```

The resulting immutable job configuration passes the same flag to the container. This is a diagnostic configuration, not launch authorization: image publication, any required immutable staging, and the paid A100 CustomJob still require their separate explicit confirmations. Use a new run and job ID so its output and launch records cannot collide with earlier attempts.

The local Linux/amd64 container check must import the full training stack, report `torch.version.cuda == "12.6"`, validate the real 8+8 smoke JSONL without network access, and fail explicitly at `cuda_preflight` when deliberately run without GPU passthrough. Passing these checks does not replace a paid GPU smoke run; it only makes that run technically reviewable.

## Preparation, staging, and image workflow

Prepare the deterministic local run from a clean committed checkout:

```bash
uv run edugraph-classify run prepare \
  --config experiments/edugraph-20260823-qwen35-4b-vertex-smoke-v1.json
```

Preparation reads the public Hugging Face dataset release and does not use GCP. Inspect `runs/edugraph-20260823-qwen35-4b-vertex-smoke-v1/manifest.json` before staging.

Staging is an explicit dataset upload and requires the exact run ID:

```bash
uv run edugraph-classify vertex stage \
  --manifest runs/edugraph-20260823-qwen35-4b-vertex-smoke-v1/manifest.json \
  --bucket-uri gs://edugraph-classify \
  --confirm-run-id edugraph-20260823-qwen35-4b-vertex-smoke-v1
```

The staging plan verifies every local SHA-256 before upload. Object names include the clean code commit and content hash; create-only generation preconditions prevent overwrite. A retry accepts an existing object only when its size and `edugraph-sha256` metadata match. The resulting runtime manifest pins the prepared manifest, dataset splits, ontology, prompt, schema, model revision, recipe, provider identity, and regional output prefix.

Build the trainer from the repository root, tag it with the clean commit, authenticate Docker to the regional Artifact Registry, and push it. Resolve the uploaded manifest to an immutable digest; never put a tag in a job configuration.

```bash
docker build -f containers/vertex-trainer/Dockerfile \
  -t europe-west4-docker.pkg.dev/edugraph-438718/training/vlm:<code-commit> .
```

After staging and image publication, generate the fixed first-smoke job configuration offline:

```bash
uv run edugraph-classify vertex configure \
  --manifest runs/edugraph-20260823-qwen35-4b-vertex-smoke-v1/manifest.json \
  --staging-record runs/edugraph-20260823-qwen35-4b-vertex-smoke-v1/vertex-staging-record.json \
  --image-uri europe-west4-docker.pkg.dev/edugraph-438718/training/vlm@sha256:<digest> \
  --output runs/edugraph-20260823-qwen35-4b-vertex-smoke-v1/vertex-job.json
```

Then render and review the exact CustomJob request. Submission remains blocked until the local container smoke succeeds, the image digest and request are reviewed, a current cost bound is recorded, and the exact job ID is explicitly confirmed. A failed or superseded image must never be reused merely because it remains present in Artifact Registry.

Official references: [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B), [Transformers Qwen3.5 architecture](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [PEFT QLoRA guidance](https://huggingface.co/docs/peft/developer_guides/quantization), [create a serverless custom job](https://cloud.google.com/vertex-ai/docs/training/create-custom-job), [configure compute](https://cloud.google.com/vertex-ai/docs/training/configure-compute), [Vertex AI locations](https://cloud.google.com/vertex-ai/docs/general/locations), and [Flex Start scheduling](https://cloud.google.com/vertex-ai/docs/training/schedule-jobs-dws).
