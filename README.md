# EduGraph Classify

Training and evaluation tooling for direct ontology labeling of atomic educational tasks with vision-language models.

The next K–6 training candidate uses the same **Qwen3.8-27B NF4 QLoRA / Runpod Secure** policy with dataset **v0.31.0-01**, ontology **v0.31.0**, all **2,330 training images**, and a **4,096-token context**. Only language adapters train; vision and the multimodal bridge stay frozen. This candidate has not yet trained or been promoted.

The selected training checkpoint used **Qwen3.8-27B NF4 QLoRA on Runpod Secure**, with **W&B** tracking and **GCS** checkpoints, dataset **v0.30.0-03**, and ontology **v0.30.0**. Only language adapters trained; vision and the multimodal bridge stayed frozen. Its compact v3 system prompt omits ontology vocabulary; a separate closed-vocabulary decoding schema and offline validation enforce the output contract. Generated predictions determined checkpoint selection and final validation. The completed two-epoch Secure A40 run achieved **72.4% exact-set match (181/250)** and **96.44% label F1** on the separate final assessment cohort, with zero invalid outputs. Results and recovery artifacts were independently verified. Production Imagine serves the **derived Q4_K_M GGUF** through the default 27B backend and predictor routes; the NF4 checkpoint is the training source, not the serving format.

See [Runpod training](docs/RUNPOD-TRAINING.md) for the supported training workflow, secrets, launch boundaries and recovery. The [full-run record](docs/RUNPOD-FULL-A40-20261002.md) preserves the exact configuration and results; [training costs](docs/TRAINING-COSTS.md) preserves the dated provider comparison. The [refinement plan](docs/REFINEMENT-PLAN.md) prioritizes label conventions, controlled contrasts and fresh evaluation; its [first audit](docs/LABEL-CONVENTION-AUDIT-20261003.md) separates model errors from unresolved annotation choices. [Project design](docs/PROJECT-KICKOFF.md) retains accepted architecture and historical decisions. This repository owns classifier conversion, prompts, orchestration, inference, evaluation, and experiment records; upstream projects own images/splits and ontology semantics.

The next dataset priorities are **consistent realized-task ranges** and **complete coverage of legitimate, observable labels**. The user deferred those changes and fresh evaluation coverage until after this K–6 training round. Its assessment is regression evidence because it reuses previously inspected images. The [upstream annotation handoff](docs/UPSTREAM-ANNOTATION-HANDOFF-20261003.md) retains the implementation boundaries and acceptance checks.

The optional [Runpod inference benchmark](docs/RUNPOD-INFERENCE-BENCHMARK.md) measures the merged-BF16 checkpoint and the Q4_K_M GGUF route for llama.cpp on the original validation cohort. The [BF16 run record](docs/RUNPOD-BF16-BENCHMARK-20261005.md) reports 40/64 exact in both A100 repeats after an initial failed warmup attempt. The [L40S GGUF record](docs/RUNPOD-GGUF-BENCHMARK-20261004.md) reports 37/64 after correcting JSON property order, versus 43/64 for the selected NF4 training model on the same 64 cases. The classification-only web release passed hosted preview and live uploads. On 2026-10-22, the default backend was replaced by a cleaned 27B-only revision that calls the untagged private 27B predictor; both the built-in and shared validation image passed at its tag and default route. The new web bundle now calls the untagged public API. The exact tested Firebase preview version was released to live, and hosted Chrome uploads of both images passed without browser errors. Guarded cleanup and an independent readback verified removal of 36 4B-client backend revisions and 96 associated OCI manifests, five 4B predictor revisions and 20 package manifests, the old model-storage prefixes' 179 live objects, and 15 obsolete Hosting versions. Gemini-era backend history and the 27B release are retained. See the [cleanup record](docs/GCP-CLEANUP.md) for the evidence boundary, GCS soft-deletion period, and storage-accounting caveat.

The [Qwen3.8-27B Q4_K_M GGUF release](docs/HUGGINGFACE-GGUF-RELEASE.md) is public under Apache-2.0 at a verified immutable Hugging Face revision. The [production baseline](docs/IMAGINE-BASELINE-PROMOTION.md) links the [historical tagged rollout](docs/IMAGINE-BASELINE-PROMOTION-20261007.md) and [default-route cutover receipt](experiments/imagine-27b-default-cutover-20261022.json), which pins the final 27B routes, live web version, and confirmed 4B cleanup.

For the reusable path from a selected training checkpoint through matched validation, model publication, a tagged GCP canary and blue-green web cutover, see [Model to production](docs/MODEL-TO-PRODUCTION.md).

## Development

Python 3.12 and [uv](https://docs.astral.sh/uv/) manage environments, dependencies, execution, locking, and builds.

```powershell
uv sync --locked --group training
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

The [K–6 recipe](experiments/edugraph-20261024-qwen38-27b-runpod-k6-v1.json) retains the successful A40 model/training policy. `runpod new-run` creates subsequent recipes from new release pins and an explicit runtime/budget; full training and remaining-validation cohorts adapt to release size. `prepare --dry-run` checks public inputs locally before committing; launchable preparation requires a clean commit. `status`, `retrieve`, and offline `verify-result` work from each run's manifest, without run-specific scripts. The [Runpod workflow](docs/RUNPOD-TRAINING.md) documents these commands and separates GCS staging, request review and paid Pod creation.

W&B uses entity `edugraph-io`, project `edugraph-classify`, and the user's Runpod secret `WANDB_API_KEY`. The training-only GCS credential is configured in Runpod Secrets. The [Secure A40 smoke](docs/RUNPOD-SMOKE-20261002.md) and [L40S benchmark](docs/RUNPOD-HARDWARE-BENCHMARK.md) verified the earlier recipe's learning, live recovery and longest inputs. The [full two-epoch A40 run](docs/RUNPOD-FULL-A40-20261002.md) completed on October 3 in **5 hours 19 minutes** for approximately **$2.73** of Runpod spend. All three Pods terminated automatically after publishing their artifacts; these measurements do not establish GPU fit or runtime for the larger K–6 inputs.

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

This project's code is licensed under the [Apache License 2.0](LICENSE). New EduGraph model-weight releases target Apache-2.0 after verifying the exact upstream checkpoint's redistribution terms; see the [GGUF release workflow](docs/HUGGINGFACE-GGUF-RELEASE.md). Upstream datasets, ontology content, and bundled dependencies retain their own licenses.
