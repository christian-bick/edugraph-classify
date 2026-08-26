# EduGraph Classify

Training and evaluation tooling for direct ontology labeling of atomic educational tasks with vision-language models.

The first project milestone is a direct-labeling VLM baseline. Fireworks was tested first and rejected the selected Qwen3-VL checkpoint at job creation; Google Vertex AI serverless custom training is now the second managed-training path. Provider adapters keep data preparation and evaluation independent from provider APIs, while execution mode remains explicit across serverless, provider-dedicated, and self-hosted paths. The repository does not own dataset generation or ontology authoring.

The accepted architecture and experiment rationale are recorded in [Project kickoff and baseline decision](docs/PROJECT-KICKOFF.md).

## Status

Provider-neutral identities, dimension-aware conversion, pinned ontology validation, deterministic VLM JSONL generation, guarded managed-training operations, Fireworks model preflight, and a Vertex AI CustomJob adapter are in place. The Fireworks technical smoke datasets reached `READY`, but Fireworks rejected Qwen3-VL-8B-Instruct at managed-job creation as unsupported. Vertex bootstrap includes a dedicated keyless runtime identity, regional Artifact Registry repository, bucket-scoped IAM, immutable GCS staging, and a containerized QLoRA trainer for the pinned public 2026 `Qwen/Qwen3.5-4B` VLM revision. L4 Flex Start attempts exhausted their two-hour capacity wait. A100 attempts established working provisioning, CUDA, model loading, quantized preparation, and LoRA attachment; a first-batch failure caused by text-only prompt masking was corrected by processing the mask prefix through the same multimodal path. The corrected run cleared that invariant and entered the first model training step, where it exposed a distinct runtime failure that still requires bounded diagnostics. No trained model has been produced yet.

## Development

The project requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for environments, dependency locking, command execution, and builds.

```bash
uv sync
uv run pytest --cov=edugraph_classify --cov-report=term-missing
uv build
```

The initial runtime dependencies are:

- `fireworks-ai`, the official Fireworks Python SDK for inference and platform orchestration;
- `google-cloud-aiplatform`, the official Vertex AI SDK used only by the GCP provider boundary;
- `google-cloud-storage`, used for immutable, hash-checked run staging and artifact transfer;
- `datasets`, the Hugging Face library used to access and process the released image dataset;
- `fsspec` and `pillow`, used to read and validate the pinned release images deterministically;
- `edugraph-py`, installed from the official `v0.21.0` wheel and used as the ontology authority;
- `python-dotenv`, used only at application entry points to load local configuration without overriding process-level environment variables.

`uv.lock` is committed. Add or update dependencies with `uv add`/`uv remove` and commit the resulting `pyproject.toml` and lockfile together.

The initial baseline data configuration pins `christian-bick/edugraph-exercises` tag `v0.21.0-01` at commit `ed47264f751a8a67c480dbe4eb7397e317a722eb` with ontology release `v0.21.0`. Dataset ingestion uses `datasets` directly rather than a provider adapter.

## Provider preflight and first-run preparation

Inspect the exact Fireworks model selected for the first run:

```bash
uv run edugraph-classify preflight fireworks
```

The command loads `.env` explicitly, preserves any variables already supplied by the process, and never prints secret values. An ineligible model produces structured diagnostic output and exit code `2`. The preflight is read-only; it does not launch training, create a deployment, upload data, or publish a model.

Create deterministic provider-ready artifacts from the pinned public Hugging Face release:

```bash
uv run edugraph-classify run prepare
```

Preparation requires a clean committed worktree and writes its manifest, source audit data, profiles, and JSONL files under the gitignored `runs/` directory. It does not need a Hugging Face token for the public release and does not call Fireworks.

Launching is a separate, cost-incurring command that rechecks the clean code commit, artifact hashes, exact live model eligibility, and resource-name collisions. It requires the prepared run ID as explicit confirmation:

```bash
uv run edugraph-classify run launch \
  --manifest runs/edugraph-20260823-qwen3vl8b-sft-v1/manifest.json \
  --confirm-run-id edugraph-20260823-qwen3vl8b-sft-v1
```

The launch creates and validates two Fireworks datasets and starts one supervised LoRA job. It records provider identifiers in `launch-record.json`; it does not create a deployment or publish the resulting model.

The full pinned release currently fails the required cross-split image-byte safeguard. A separate tracked eight-example-per-split smoke configuration exists solely to verify managed-training plumbing and must not be treated as a quality baseline:

```bash
uv run edugraph-classify run prepare \
  --config experiments/edugraph-20260823-qwen3vl8b-smoke-v1.json
```

## Vertex AI serverless custom training

Vertex AI CustomJob is treated as `serverless` training because Google owns job provisioning and teardown, even though each run still declares its ephemeral machine and GPU shape. The adapter uses the regional API endpoint from the job configuration and supports standard, Spot, and Flex Start scheduling.

Validate and inspect the exact provider request without credentials or network access:

```bash
uv run edugraph-classify vertex render --config runs/<job-id>/vertex-job.json
```

Submitting a job is separate and cost-incurring. It requires Application Default Credentials, a clean checkout at the commit pinned in the configuration, an immutable training-image digest, and the exact job ID as confirmation:

```bash
uv run edugraph-classify vertex launch \
  --config runs/<job-id>/vertex-job.json \
  --confirm-job-id <job-id>
```

The first tracked Vertex smoke recipe is `experiments/edugraph-20260823-qwen35-4b-vertex-smoke-v1.json`. Preparation, content-addressed staging, immutable image publication, offline job configuration/rendering, and paid submission are deliberately separate operations. The setup, commands, IAM boundary, model/training decision, strict configuration shape, and EU-region guidance are documented in [Vertex AI serverless training](docs/VERTEX-AI-TRAINING.md).

## Related projects

- [edugraph-dataset](https://github.com/christian-bick/edugraph-dataset): canonical labeled image dataset releases
- [edugraph-ontology](https://github.com/christian-bick/edugraph-ontology): ontology entities, dimensions, definitions, and relations
- [edugraph-classify-qwen3vl](https://github.com/christian-bick/edugraph-classify-qwen3vl): earlier Qwen3-VL classifier experiment and checkpoint lineage
