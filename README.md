# EduGraph Classify

Training and evaluation tooling for direct ontology labeling of atomic educational tasks with vision-language models.

The first project milestone is a direct-labeling baseline using Qwen3.5-9B as the starting model hypothesis and Fireworks AI as the initial managed provider. Managed-training adapters keep data preparation and evaluation independent from provider APIs, while execution mode remains explicit across serverless, provider-dedicated, and self-hosted paths. The repository will not own dataset generation or ontology authoring.

The accepted architecture and experiment rationale are recorded in [Project kickoff and baseline decision](docs/PROJECT-KICKOFF.md).

## Status

The initial provider-neutral identities, dimension-aware prediction contract, managed-training provider boundary, and read-only Fireworks model preflight are in place. No training job has been launched and no production classifier has been selected.

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
- `python-dotenv`, used only at application entry points to load local configuration without overriding process-level environment variables.

`uv.lock` is committed. Add or update dependencies with `uv add`/`uv remove` and commit the resulting `pyproject.toml` and lockfile together.

The initial baseline data configuration pins `christian-bick/edugraph-exercises` tag `v0.21.0-01` at commit `ed47264f751a8a67c480dbe4eb7397e317a722eb` with ontology release `v0.21.0`. Dataset ingestion uses `datasets` directly rather than a provider adapter.

## Read-only provider preflight

Inspect the exact Fireworks model behind the Qwen3.5-9B hypothesis:

```bash
uv run edugraph-classify preflight fireworks
```

The command loads `.env` explicitly, preserves any variables already supplied by the process, and never prints secret values. An ineligible model produces structured diagnostic output and exit code `2`. The preflight is read-only; it does not launch training, create a deployment, upload data, or publish a model.

## Related projects

- [edugraph-dataset](https://github.com/christian-bick/edugraph-dataset): canonical labeled image dataset releases
- [edugraph-ontology](https://github.com/christian-bick/edugraph-ontology): ontology entities, dimensions, definitions, and relations
- [edugraph-classify-qwen3vl](https://github.com/christian-bick/edugraph-classify-qwen3vl): earlier Qwen3-VL classifier experiment and checkpoint lineage
