# EduGraph Classify

Training and evaluation tooling for direct ontology labeling of atomic educational tasks with vision-language models.

The first project milestone is a direct-labeling baseline using Qwen3-VL-8B-Instruct and Fireworks AI. Managed-training adapters keep data preparation and evaluation independent from provider APIs, while execution mode remains explicit across serverless, provider-dedicated, and self-hosted paths. The repository does not own dataset generation or ontology authoring.

The accepted architecture and experiment rationale are recorded in [Project kickoff and baseline decision](docs/PROJECT-KICKOFF.md).

## Status

Provider-neutral identities, dimension-aware conversion, pinned ontology validation, deterministic Fireworks VLM JSONL generation, guarded managed-training operations, and read-only model preflight are in place. The technical smoke datasets reached `READY`, but Fireworks rejected Qwen3-VL-8B-Instruct at managed-job creation as unsupported. No training job, output model, or deployment was created.

## Development

The project requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for environments, dependency locking, command execution, and builds.

```bash
uv sync
uv run pytest --cov=edugraph_classify --cov-report=term-missing
uv build
```

The initial runtime dependencies are:

- `fireworks-ai`, the official Fireworks Python SDK for inference and platform orchestration;
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

## Related projects

- [edugraph-dataset](https://github.com/christian-bick/edugraph-dataset): canonical labeled image dataset releases
- [edugraph-ontology](https://github.com/christian-bick/edugraph-ontology): ontology entities, dimensions, definitions, and relations
- [edugraph-classify-qwen3vl](https://github.com/christian-bick/edugraph-classify-qwen3vl): earlier Qwen3-VL classifier experiment and checkpoint lineage
