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

## Resolved job configuration

Generated job configurations belong under `runs/<job-id>/` and remain gitignored. They contain reproducibility metadata and infrastructure identifiers, but no secrets. The strict JSON shape is:

```json
{
  "provider_id": "gcp_vertex_ai",
  "training_execution_mode": "serverless",
  "job_id": "edugraph-vertex-smoke",
  "code_commit": "0123456789abcdef0123456789abcdef01234567",
  "project_id": "edugraph-prod",
  "location": "europe-west4",
  "display_name": "EduGraph Vertex smoke",
  "output_uri": "gs://edugraph-training/runs/edugraph-vertex-smoke",
  "service_account": "vertex-training@edugraph-prod.iam.gserviceaccount.com",
  "container": {
    "image_uri": "europe-west4-docker.pkg.dev/edugraph-prod/training/vlm@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "command": ["python", "-m", "edugraph_trainer"],
    "args": ["--run-manifest", "gs://edugraph-training/runs/edugraph-vertex-smoke/manifest.json"],
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

## Required next implementation

The adapter alone does not make a model trainable. Before the first Vertex smoke job, the repository still needs:

1. a provider-neutral VLM trainer entry point and container;
2. an immutable 2026 model repository and revision that fits the 3B–12B target;
3. a deterministic staging step for the prepared run manifest and JSONL inputs;
4. a model-specific GPU-memory estimate and EU-region machine choice;
5. a local container smoke test and a rendered Vertex request review;
6. explicit approval for the estimated billable training run.

Official references: [create a serverless custom job](https://cloud.google.com/vertex-ai/docs/training/create-custom-job), [configure compute](https://cloud.google.com/vertex-ai/docs/training/configure-compute), [Vertex AI locations](https://cloud.google.com/vertex-ai/docs/general/locations), and [Flex Start scheduling](https://cloud.google.com/vertex-ai/docs/training/schedule-jobs-dws).
