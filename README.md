# EduGraph Classify

Training and evaluation tooling for direct ontology labeling of atomic educational tasks with vision-language models.

The current full-training candidate uses **Qwen3.8-27B on Fireworks serverless training**, dataset **v0.30.0-03**, and ontology **v0.30.0**. It is unlaunched. The earlier v0.30.0-02 learning smoke stopped after one epoch on 160 training images: label F1 on 64 official-validation images rose from 30.9% to 67.4%. Exact-set match remained 2/64, so the private checkpoint is experimental. Training and inference use the same fixed system prompt.

See [Training API setup](docs/FIREWORKS-TRAINING-API.md) for the recipe, evaluation policy, system-prompt template, provider limits, and lifecycle. [Project kickoff](docs/PROJECT-KICKOFF.md) retains accepted architecture and historical decisions. This repository owns classifier conversion, prompts, orchestration, inference, evaluation, and experiment records; upstream projects own images/splits and ontology semantics.

## Development

Python 3.12 and [uv](https://docs.astral.sh/uv/) manage environments, dependencies, execution, locking, and builds.

```powershell
uv sync --group training-api
uv run --group training-api pytest --cov=edugraph_classify --cov-report=term-missing
uv build
```

`uv.lock` is committed. Keep it synchronized with `pyproject.toml`. The optional `training-api` group adds the official training SDK and local tokenizer/image processing tools. Windows uses CPU PyTorch; existing Linux training containers retain CUDA 12.6. Provider calls remain behind thin adapters and unit tests use fakes.

Credentials belong in environment variables or the gitignored `.env`; the CLI loads it without overriding process variables. Generated datasets, logs, raw predictions, checkpoints, and reports belong under gitignored `data/`, `artifacts/`, `runs/`, `reports/`, or `temp/`.

## Prepare the full-training candidate

```powershell
uv run --group training-api edugraph-classify training-api prepare `
  --config experiments/edugraph-20260929-qwen38-27b-full-v2.json
uv run --group training-api edugraph-classify training-api launch `
  --manifest runs/edugraph-20260929-qwen38-27b-full-v2/manifest.json `
  --confirm-run-id edugraph-20260929-qwen38-27b-full-v2
```

Preparation performs a read-only live eligibility check and local conversion of public data. It requires a clean commit and pins all identities and artifact hashes. Launch is separate, explicitly confirmed, and cost-incurring. It runs a bounded paired evaluation/training experiment, saves state each epoch, retains the final private adapter, and closes the pooled session. An execution journal blocks accidental paid replay. It does not create an inference deployment.

A completed run can be continued from a saved training-state checkpoint in a separately prepared run with a new ID. Use the local `training-api resume-check` before the paid `training-api resume` command; see the [continuation workflow](docs/FIREWORKS-TRAINING-API.md#commands-and-lifecycle). Resumption restores optimizer state and starts at the next epoch. The v0.30.0-02 full-release candidate exposed a conflicting duplicate pair; the v0.30.0-03 candidate keeps its own immutable recipe and must pass local full-release preparation before any paid run.

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
