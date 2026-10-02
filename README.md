# EduGraph Classify

Training and evaluation tooling for direct ontology labeling of atomic educational tasks with vision-language models.

The current training candidate uses **Qwen3.8-27B NF4 QLoRA on Runpod Secure**, with **W&B** tracking and **GCS** checkpoints, dataset **v0.30.0-03**, and ontology **v0.30.0**. Only language adapters train; vision and the multimodal bridge stay frozen. Its compact v3 system prompt omits ontology vocabulary; a separate closed-vocabulary decoding schema and offline validation enforce the output contract. Generated predictions determine checkpoint selection and final validation. The Secure A40 smoke improved held-out label F1 from **3.18% to 61.25%** after one epoch on 160 images. Exact-set match remained **0/64**, so the checkpoint is experimental.

See [Runpod training](docs/RUNPOD-TRAINING.md) for setup, secrets, launch boundaries and recovery; [Fireworks Training API](docs/FIREWORKS-TRAINING-API.md) for the preserved managed executor; and [training costs](docs/TRAINING-COSTS.md) for the compact-prompt comparison. [Project kickoff](docs/PROJECT-KICKOFF.md) retains accepted architecture and historical decisions. This repository owns classifier conversion, prompts, orchestration, inference, evaluation, and experiment records; upstream projects own images/splits and ontology semantics.

## Development

Python 3.12 and [uv](https://docs.astral.sh/uv/) manage environments, dependencies, execution, locking, and builds.

```powershell
uv sync --group training --group training-api
uv run --no-sync pytest --cov=edugraph_classify --cov-report=term-missing
uv build
```

`uv.lock` is committed. Keep it synchronized with `pyproject.toml`. The optional `training-api` group adds the official training SDK and local tokenizer/image processing tools. Windows uses CPU PyTorch; existing Linux training containers retain CUDA 12.6. Provider calls remain behind thin adapters and unit tests use fakes.

Credentials belong in environment variables or the gitignored `.env`; the CLI loads it without overriding process variables. Generated datasets, logs, raw predictions, checkpoints, and reports belong under gitignored `data/`, `artifacts/`, `runs/`, `reports/`, or `temp/`.

New self-hosted training images use **GitHub Container Registry**. The manual [container publishing workflow](docs/CONTAINER-IMAGES.md) builds the locked Linux/CUDA environment, checks it without a GPU, and optionally publishes a digest-pinned image. GCS continues to hold run artifacts, and production inference remains on GCP with scale-to-zero. The [GCP cleanup record](docs/GCP-CLEANUP.md) identifies deleted training registries and retained production resources.

## Prepare the Runpod candidate

```powershell
uv run --group training edugraph-classify runpod check-config
uv run --group training edugraph-classify runpod prepare `
  --config experiments/edugraph-20261002-qwen38-27b-runpod-smoke-v2.json
```

Preparation requires a clean commit and downloads public inputs locally. The [Runpod workflow](docs/RUNPOD-TRAINING.md) separates GCS staging, request review and paid Pod creation. W&B uses entity `edugraph-io`, project `edugraph-classify`, and the user's Runpod secret `WANDB_API_KEY`. The training-only GCS credential is configured in Runpod Secrets. The [Secure A40 smoke](docs/RUNPOD-SMOKE-20261002.md) completed, published its model and resumable state, and terminated its Pod. Full training needs a separate decision.

## Preserved Fireworks candidate

```powershell
uv run --group training-api edugraph-classify training-api prepare `
  --config experiments/edugraph-20261002-qwen38-27b-full-v3.json
uv run --group training-api edugraph-classify training-api launch `
  --manifest runs/edugraph-20261002-qwen38-27b-full-v3/manifest.json `
  --confirm-run-id edugraph-20261002-qwen38-27b-full-v3
```

Preparation performs a read-only live eligibility check and local conversion of public data. It requires a clean commit and pins all identities and artifact hashes. Launch is separate, explicitly confirmed, and cost-incurring. It runs a bounded paired evaluation/training experiment, saves state each epoch, retains the final private adapter, and closes the pooled session. An execution journal blocks accidental paid replay. It does not create an inference deployment.

A completed run can be continued from a saved training-state checkpoint in a separately prepared run with a new ID. Use the local `training-api resume-check` before the paid `training-api resume` command; see the [continuation workflow](docs/FIREWORKS-TRAINING-API.md#commands-and-lifecycle). Resumption restores optimizer state and starts at the next epoch. The compact v3 prompt requires a fresh run; a v2-prompt checkpoint is incompatible with that continuation contract. A local v0.30.0-03 audit verified all 1,968 images without byte duplicates or split overlap; clean-commit preparation remains required before launch.

The system prompt is identical across training and inference, with assistant-only loss. Preparation also exports a locally verified template that embeds the prompt. Fireworks currently documents custom chat templates for base models, not LoRA adapters; see [the deployment-template limitation](docs/FIREWORKS-TRAINING-API.md#fixed-system-prompt-and-deployment-template).

## Historical paths

Historical recipes retain their original dataset, ontology, model, and environment pins. Reproduce them from their original code commit and lockfile rather than silently upgrading their gold labels.

- [Local Docker training](docs/LOCAL-TRAINING.md): planned Qwen3.5-9B DDP/QLoRA on two RTX 3090s, with GCP artifact services. Preparation exists; the distributed launch adapter remains planned.
- [Vertex AI training](docs/VERTEX-AI-TRAINING.md): custom-job provisioning, immutable staging, and a containerized Qwen3.5-4B trainer. Earlier A100 diagnostics isolated a PyTorch native-JIT rotary-position failure.
- Fireworks managed SFT: earlier Qwen3-VL-8B datasets reached READY, but the provider rejected that model at job creation. The newer serverless training API is a separate execution path.
- [Dataset v0.26.0-01 audit](docs/DATASET-UPGRADE-0.26.0-01.md) and [ontology v0.27 adoption](docs/ONTOLOGY-UPGRADE-0.27.md): historical full-release audit and migration evidence.

## Upstream contracts

- [Dataset contract](docs/UPSTREAM-DATASET.md): released images, explicit gold labels, official splits, and atomic classification units.
- [Ontology contract](docs/UPSTREAM-ONTOLOGY.md): identifiers, dimensions, eligibility, and explicit-versus-derived semantics.
- [edugraph-dataset](https://github.com/christian-bick/edugraph-dataset)
- [edugraph-ontology](https://github.com/christian-bick/edugraph-ontology)
- [Earlier Qwen3-VL classifier](https://github.com/christian-bick/edugraph-classify-qwen3vl)

## License

This project's code is licensed under the [Apache License 2.0](LICENSE). Upstream datasets, ontology content, model weights, and bundled dependencies retain their own licenses.
