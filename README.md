# EduGraph Classify

Training and evaluation tooling for direct ontology labeling of atomic educational tasks with vision-language models.

The first project milestone is a direct-labeling baseline using Qwen3.5-9B as the starting model hypothesis and Fireworks AI as the initial managed provider. Provider adapters keep data preparation and evaluation independent from provider APIs and leave room for both serverless and self-hosted execution. The repository will not own dataset generation or ontology authoring.

The accepted architecture and experiment rationale are recorded in [Project kickoff and baseline decision](docs/PROJECT-KICKOFF.md).

## Status

The initial Python package and locked provider/data dependencies are in place. No training job has been launched and no production classifier has been selected.

## Development

The project requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for environments, dependency locking, command execution, and builds.

```bash
uv sync
uv build
```

The initial runtime dependencies are:

- `fireworks-ai`, the official Fireworks Python SDK for inference and platform orchestration;
- `datasets`, the Hugging Face library used to access and process the released image dataset.

`uv.lock` is committed. Add or update dependencies with `uv add`/`uv remove` and commit the resulting `pyproject.toml` and lockfile together.

## Related projects

- [edugraph-dataset](https://github.com/christian-bick/edugraph-dataset): canonical labeled image dataset releases
- [edugraph-ontology](https://github.com/christian-bick/edugraph-ontology): ontology entities, dimensions, definitions, and relations
- [edugraph-classify-qwen3vl](https://github.com/christian-bick/edugraph-classify-qwen3vl): earlier Qwen3-VL classifier experiment and checkpoint lineage
