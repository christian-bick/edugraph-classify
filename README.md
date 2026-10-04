# EduGraph Classify

Training and evaluation tooling for direct ontology labeling of atomic educational tasks with vision-language models.

The current training candidate uses **Qwen3.8-27B NF4 QLoRA on Runpod Secure**, with **W&B** tracking and **GCS** checkpoints, dataset **v0.30.0-03**, and ontology **v0.30.0**. Only language adapters train; vision and the multimodal bridge stay frozen. Its compact v3 system prompt omits ontology vocabulary; a separate closed-vocabulary decoding schema and offline validation enforce the output contract. Generated predictions determine checkpoint selection and final validation. The completed two-epoch Secure A40 run achieved **72.4% exact-set match (181/250)** and **96.44% label F1** on the separate final assessment cohort, with zero invalid outputs. Results and recovery artifacts were independently verified; the model has not been promoted or deployed.

See [Runpod training](docs/RUNPOD-TRAINING.md) for the supported training workflow, secrets, launch boundaries and recovery. The [full-run record](docs/RUNPOD-FULL-A40-20261002.md) preserves the exact configuration and results; [training costs](docs/TRAINING-COSTS.md) preserves the dated provider comparison. The [refinement plan](docs/REFINEMENT-PLAN.md) prioritizes label conventions, controlled contrasts and fresh evaluation; its [first audit](docs/LABEL-CONVENTION-AUDIT-20261003.md) separates model errors from unresolved annotation choices. [Project design](docs/PROJECT-KICKOFF.md) retains accepted architecture and historical decisions. This repository owns classifier conversion, prompts, orchestration, inference, evaluation, and experiment records; upstream projects own images/splits and ontology semantics.

The next dataset priorities are **consistent realized-task ranges** and **complete coverage of legitimate, observable labels**. The [upstream annotation handoff](docs/UPSTREAM-ANNOTATION-HANDOFF-20261003.md) traces generator/view causes and specifies implementation boundaries, examples and acceptance checks for both changes.

The optional [Runpod inference benchmark](docs/RUNPOD-INFERENCE-BENCHMARK.md) merges the selected adapter, exports Q4_K_M GGUF for llama.cpp, and measures accuracy, latency and GPU fit on the original validation cohort. The [first completed L40S run](docs/RUNPOD-GGUF-BENCHMARK-20261004.md) scored 19/64 exact after export, versus 43/64 for the selected NF4 training model on the same cohort; it was not promoted. This workflow is separate from training and does not change the Imagine deployment.

## Development

Python 3.12 and [uv](https://docs.astral.sh/uv/) manage environments, dependencies, execution, locking, and builds.

```powershell
uv sync --group training
uv run --no-sync pytest --cov=edugraph_classify --cov-report=term-missing
uv build
```

`uv.lock` is committed. Keep it synchronized with `pyproject.toml`. The `training` group contains the Runpod worker's model and evaluation dependencies. Windows uses CPU PyTorch; Linux training containers use CUDA 12.6. Provider calls remain behind thin adapters and unit tests use fakes.

Credentials belong in environment variables or the gitignored `.env`; the CLI loads it without overriding process variables. Generated datasets, logs, raw predictions, checkpoints, and reports belong under gitignored `data/`, `artifacts/`, `runs/`, `reports/`, or `temp/`.

Runpod training images use **GitHub Container Registry**. The manual [container publishing workflow](docs/CONTAINER-IMAGES.md) builds the locked Linux/CUDA environment with the Runpod worker as its default entrypoint, checks it without a GPU, and optionally publishes a digest-pinned image. GCS continues to hold run artifacts, and production inference remains on GCP with scale-to-zero. The [GCP cleanup record](docs/GCP-CLEANUP.md) identifies deleted training registries and retained production resources.

## Prepare a Runpod experiment

```powershell
uv run --group training edugraph-classify runpod check-config
```

For a new experiment, copy and review the [successful A40 recipe](experiments/edugraph-20261002-qwen38-27b-runpod-full-a40-v1.json), assign a new run ID, then prepare it with `uv run --group training edugraph-classify runpod prepare --config <recipe-path>`. Preparation requires a clean commit and downloads public inputs locally. The [Runpod workflow](docs/RUNPOD-TRAINING.md) separates GCS staging, request review and paid Pod creation. W&B uses entity `edugraph-io`, project `edugraph-classify`, and the user's Runpod secret `WANDB_API_KEY`. The training-only GCS credential is configured in Runpod Secrets. The [Secure A40 smoke](docs/RUNPOD-SMOKE-20261002.md) and [L40S benchmark](docs/RUNPOD-HARDWARE-BENCHMARK.md) verified learning, live checkpoint recovery and longest-input fit. The [full two-epoch A40 run](docs/RUNPOD-FULL-A40-20261002.md) completed on October 3 in **5 hours 19 minutes** for approximately **$2.73** of Runpod spend. All three Pods terminated automatically after publishing their artifacts.

## Historical paths

The following records describe retired training paths. Their executors, recipes, old prompts and dependency setup are absent from the current checkout. Reproduce them only from their original code commit and lockfile, with the original dataset, ontology, model and environment pins.

- [Fireworks Training API](docs/FIREWORKS-TRAINING-API.md): a 27B learning smoke and an unlaunched full-run candidate, retained for provenance and comparison.
- [Local Docker training](docs/LOCAL-TRAINING.md): proposed Qwen3.5-9B DDP/QLoRA on two RTX 3090s, with GCP artifact services.
- [Vertex AI training](docs/VERTEX-AI-TRAINING.md): historical custom-job setup and Qwen3.5-4B diagnostics. Earlier A100 diagnostics isolated a PyTorch native-JIT rotary-position failure.
- Fireworks managed SFT: earlier Qwen3-VL-8B datasets reached READY, but the provider rejected that model at job creation.
- [Dataset v0.26.0-01 audit](docs/DATASET-UPGRADE-0.26.0-01.md) and [ontology v0.27 adoption](docs/ONTOLOGY-UPGRADE-0.27.md): historical full-release audit and migration evidence.

## Upstream contracts

- [Dataset contract](docs/UPSTREAM-DATASET.md): released images, explicit gold labels, official splits, and atomic classification units.
- [Ontology contract](docs/UPSTREAM-ONTOLOGY.md): identifiers, dimensions, eligibility, and explicit-versus-derived semantics.
- [edugraph-dataset](https://github.com/christian-bick/edugraph-dataset)
- [edugraph-ontology](https://github.com/christian-bick/edugraph-ontology)
- [Earlier Qwen3-VL classifier](https://github.com/christian-bick/edugraph-classify-qwen3vl)

## License

This project's code is licensed under the [Apache License 2.0](LICENSE). Upstream datasets, ontology content, model weights, and bundled dependencies retain their own licenses.
