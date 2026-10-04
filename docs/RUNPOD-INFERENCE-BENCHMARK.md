# Optional Runpod inference benchmark

This workflow measures whether a selected Runpod training export remains accurate and fits the intended GPU after merging and GGUF quantization. It does not deploy an inference server. Run it after a material change to the selected adapter, model revision, prompt/schema, quantization recipe, llama.cpp revision, evaluation code, or target GPU. The [completed A40 training run](RUNPOD-FULL-A40-20261002.md) is the current source candidate; its 72.4% exact-set result belongs to NF4 plus the adapter and must not be attributed to a merged or quantized export until measured.

## Pipeline and pinned inputs

```text
verified training model bundle + original prepared training manifest
    -> local preparation of the held-out validation cohort
    -> immutable staged Runpod input bundle
    -> merge the pinned BF16 base model with the selected language adapter
    -> convert merged model and vision projector to GGUF
    -> quantize language GGUF to Q4_K_M; retain the vision projector separately
    -> llama-server replay, validation, timing, and durable reports
```

The merged Hugging Face checkpoint is a local intermediate common to this GGUF route and any later vLLM or SGLang quantization route. The benchmark records its shard hashes; a later route can regenerate it from the pinned base and adapter without uploading another roughly 54 GB checkpoint. The durable model artifacts are the Q4_K_M language GGUF and BF16 multimodal projector, alongside raw responses, normalized predictions, metrics, and timings. No pretrained weights are baked into the container image.

The recipe, prepared manifest and final report pin the source model bundle URI, byte size and SHA-256; base model ID and exact revision; dataset release and revision; ontology version; prompt, schema and processor identities; code commit and `uv.lock`; llama.cpp commit; quantization mode; validation cohort; generation settings; GPU; and concurrency/repetition settings. The rendered request and launch record pin the container digest. For the current candidate, the base is `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, dataset `v0.30.0-03`, ontology `0.30.0`, and GGUF mode `Q4_K_M`. The image pins llama.cpp release `b11384` at `0faee5004297c3bcfa40b7bf11750127b8c1fd7d` and CUDA 12.6.3 base images by digest. The Docker build checks converter, quantizer and server entry points without downloading a model. A local vocabulary-only conversion of the saved Qwen3.8 processor succeeded with the frozen Python environment; full tensor conversion, the vision projector, and serving remain GPU benchmark acceptance checks.

The BF16 merge and GGUF conversion are larger than the earlier NF4 training load. Plan for **at least 100 GB host RAM and more than 200 GB of temporary disk** for the base cache, merged checkpoint, BF16 GGUF, and Q4 GGUF, then verify the live Pod allocation before launch. The worker checks visible GPU type/count/VRAM and effective RAM/CPU against the recipe before merging, and records the measured values. A training A40 Pod's 50 GB host RAM and 140 GB workspace volume are insufficient for this export. The example recipe requests an L40S, 120 GB host RAM and a 350 GB volume. Its L40S results cannot establish fit or latency on the 24 GB L4 currently used by GCP inference.

Preparation uses the original held-out **validation** cohort, with all renderings of one underlying task kept in its original split. It copies the selected export's processor, compact prompt, closed vocabulary, schema and chat template. The final assessment cohort stays reserved. Benchmark reports preserve raw output and explicit predictions alongside deterministic canonicalization and ontology validation; derived closure is not scored as an explicit model prediction. The primary result is exact-set match, supported by precision/recall/F1, per-dimension, per-view, frequency slices, invalid-output counts, startup time, peak GPU memory and cost estimates. Per-example `latency_seconds` includes image hashing, RGB JPEG conversion, request construction and the server response; `server_latency_seconds` isolates the HTTP/server roundtrip. Throughput includes all preparation and responses across the round. Runpod latency and startup measurements describe the chosen Runpod GPU and storage path; they do not establish Cloud Run cold-start behavior.

## Build an image without running a Pod

The **Build inference benchmark image** GitHub Actions workflow builds only `linux/amd64`. Once the workflow exists on the default branch, manually select a reviewed clean commit and leave **publish** unchecked for a build-only check. GitHub requires a workflow to exist on the default branch before manual dispatch; for the first run, explicitly push an `inference-benchmark-image-build-<short-commit>` tag pointing to the reviewed commit. After that build succeeds, push an `inference-benchmark-image-publish-<short-commit>` tag pointing to the **same** commit to build and publish. Ordinary branch pushes do not trigger image builds. Each pushed tag authorizes its corresponding GitHub Actions job and any applicable charges; the publish tag also authorizes a GHCR write. Never move a published tag to a different commit.

```bash
git tag inference-benchmark-image-build-<short-commit> <full-code-commit>
git push origin refs/tags/inference-benchmark-image-build-<short-commit>
# After the build-only job succeeds:
git tag inference-benchmark-image-publish-<short-commit> <full-code-commit>
git push origin refs/tags/inference-benchmark-image-publish-<short-commit>
```

A publishing job pushes `ghcr.io/christian-bick/edugraph-classify-inference-benchmark:sha-<full-code-commit>` and prints its digest URI, source commit, llama.cpp commit and lockfile SHA-256 in the workflow summary. Supply the **digest URI** to the render and launch commands; a tag alone is mutable. New GHCR packages can start private, so a public Runpod pull requires a separately configured package visibility setting or registry credential. Image publication does not create a Pod or upload a dataset.

The Docker context contains only `pyproject.toml`, `uv.lock`, `README.md`, `LICENSE`, `src/`, and the Dockerfiles. The image synchronizes the frozen `training` and `inference-benchmark` dependency groups, includes `/opt/llama.cpp/convert_hf_to_gguf.py`, `/opt/llama.cpp/build/bin/llama-quantize`, and `/opt/llama.cpp/build/bin/llama-server`, and runs offline import/help checks during the build. The source model remains an external, hash-verified input.

For a local image build with Docker and no registry publication:

```bash
docker build --platform linux/amd64 \
  -f containers/runpod-inference-benchmark/Dockerfile \
  -t edugraph-classify-inference-benchmark:local .
docker run --rm --network none \
  --entrypoint /app/.venv/bin/edugraph-classify \
  edugraph-classify-inference-benchmark:local inference-benchmark --help
```

## Prepare, review, and launch separately

Start from [the example recipe](../experiments/inference-benchmark-qwen38-27b-gguf.json), review it and assign a new run ID for a new benchmark; substitute the copied recipe path in the commands below. Local preparation needs the original prepared training manifest and a downloaded, verified selected model bundle. The source bundle is an immutable GCS object recorded in the [full-run record](RUNPOD-FULL-A40-20261002.md); its exact URI, size and SHA-256 go in the recipe. Obtain the bundle through the approved read-only recovery path and keep it in `artifacts/` or another gitignored artifact directory. Do not put model weights, credentials or generated JSONL in the repository root.

```powershell
uv sync --group training --group inference-benchmark
uv run --no-sync edugraph-classify inference-benchmark check-config --config 'experiments/inference-benchmark-qwen38-27b-gguf.json'
uv run --no-sync edugraph-classify inference-benchmark prepare `
  --config 'experiments/inference-benchmark-qwen38-27b-gguf.json' `
  --training-manifest 'runs/<training-run-id>/manifest.json' `
  --model-bundle 'artifacts/<verified-model-export>.tar'
```

Preparation, staging and request rendering require the same clean code commit. Preparation creates `runs/<benchmark-run-id>/manifest.json` with the pinned 64-image validation sample, hashed image inputs, original prompt/schema/closed vocabulary/template/processor, examples, and reduced training examples for later checks. This local step neither uploads data nor creates provider resources. Inspect the manifest, cohort, model hash, prompt/schema and expected limits before the provider steps.

**The remaining commands are separate operations.** `stage` uploads the immutable input bundle to GCS. `launch` creates a billable Runpod Secure Pod. The user must explicitly authorize each external operation and its cost before it is run. Both require the exact benchmark run ID; rendering the request is a local review step.

```powershell
$runId = '<benchmark-run-id>'
$manifest = "runs/$runId/manifest.json"
$image = 'ghcr.io/christian-bick/edugraph-classify-inference-benchmark@sha256:<published-digest>'

uv run --no-sync edugraph-classify inference-benchmark stage --manifest $manifest --confirm-run-id $runId
uv run --no-sync edugraph-classify inference-benchmark render --manifest $manifest --image $image
# Review runs/<benchmark-run-id>/pod-request.json before an authorized launch.
uv run --no-sync edugraph-classify inference-benchmark launch --manifest $manifest --image $image --confirm-run-id $runId
uv run --no-sync edugraph-classify inference-benchmark status --launch-record "runs/$runId/pod-launch.json"
uv run --no-sync edugraph-classify inference-benchmark stop --launch-record "runs/$runId/pod-launch.json" --confirm-run-id $runId
```

`stage` records `runs/<benchmark-run-id>/staged.json`; `render` writes `pod-request.json`; `launch` writes `pod-launch.json` before contacting Runpod so an ambiguous response cannot silently create a duplicate Pod. The worker checks its baked code and llama.cpp commits against the manifest, verifies staged inputs and source hashes, and writes a completion marker under the configured benchmark artifact prefix. Completed rounds are also saved as local progress; if a later benchmark step fails, the worker attempts to publish that progress and sanitized exception locations under the Pod's attempt prefix before stopping. Server logs are omitted from the failure bundle because they may contain input content. A successful benchmark terminates the Pod after durable publication; an interrupted or failed run needs status and billing reconciliation before another launch. If the stop or termination API fails, the worker emits a sanitized `shutdown_failed` event, and the Pod must be checked manually. Follow [Runpod training](RUNPOD-TRAINING.md) for the existing Secure Cloud, credential and launch boundaries.

## Interpreting the result

Compare the quantized result against the original NF4-plus-adapter report on the same validation cohort and contract. A merged BF16 export and Q4_K_M GGUF are distinct numerical configurations, so report each measured accuracy result separately. Compare warm latency at concurrency 1 and 4, startup-to-ready time, memory peak, invalid outputs and cost with their full hardware and runtime settings. Promotion to Imagine requires an independent Cloud Run load and cold-start check, plus ontology and response-contract compatibility; this benchmark does not change production traffic.
