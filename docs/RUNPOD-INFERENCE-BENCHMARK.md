# Optional Runpod inference benchmark

This workflow measures the selected Runpod training export on the original validation cohort at two distinct stages: after merging the adapter into the BF16 base model, and after GGUF conversion and Q4_K_M quantization for llama.cpp. It does not deploy an inference server. Run the relevant diagnostic after a material change to the selected adapter, model revision, prompt/schema, export recipe, evaluation code, or target GPU. The [completed A40 training run](RUNPOD-FULL-A40-20261002.md) is the current source candidate. Its selected epoch-2 NF4-plus-adapter model scored 43/64 exact on the original validation cohort. The [L40S GGUF benchmark and controlled replay](RUNPOD-GGUF-BENCHMARK-20261004.md) scored 19/64 before and 37/64 after correcting JSON property order on that same cohort. The [merged-BF16 diagnostics](RUNPOD-BF16-BENCHMARK-20261005.md) verified the same merged checkpoint; the first attempt failed before scoring, and the corrected v2 run scored 40/64 in both concurrency-one repeats. The corrected GGUF was subsequently [published and deployed](MODEL-TO-PRODUCTION.md); the BF16 diagnostic was not a serving artifact. The training run's separate 181/250 final-assessment score is not a like-for-like comparison.

## Pipeline and pinned inputs

```text
verified training model bundle + original prepared training manifest
    -> local preparation of the held-out validation cohort
    -> immutable staged Runpod input bundle
    -> merge the pinned BF16 base model with the selected language adapter
       |-> Transformers BF16 replay with the training renderer and JSON constraint
       |   -> paired validation report; no model-candidate publication
       -> convert merged model and vision projector to GGUF
           -> quantize language GGUF to Q4_K_M; retain the vision projector separately
           -> llama-server replay, validation, timing, and durable reports
```

The merged Hugging Face checkpoint is a local intermediate common to both diagnostics and any later vLLM or SGLang quantization route. The BF16 diagnostic scores it directly with the training renderer and constrained-generation path, records its shard hashes and publishes only a report. The GGUF benchmark records those hashes too, then publishes Q4_K_M language GGUF and the BF16 multimodal projector as candidate artifacts alongside its report. A later route can regenerate the merged checkpoint from the pinned base and adapter without uploading another roughly 54 GB checkpoint. No pretrained weights are baked into the container image.

The recipe, prepared manifest and final report pin the source model bundle URI, byte size and SHA-256; base model ID and exact revision; dataset release and revision; ontology version; prompt, schema and processor identities; code commit and `uv.lock`; validation cohort; generation settings; GPU; and repetition settings. The BF16 recipe also pins the expected digest of the merged checkpoint's file-hash map. The GGUF recipe instead pins the llama.cpp commit, quantization mode and concurrency settings. The rendered request and launch record pin the container digest. For the current candidate, the base is `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, dataset `v0.30.0-03`, ontology `0.30.0`, and GGUF mode `Q4_K_M`. The image pins llama.cpp release `b11384` at `0faee5004297c3bcfa40b7bf11750127b8c1fd7d` and CUDA 12.6.3 base images by digest. The Docker build checks converter, quantizer and server entry points without downloading a model. The v3 benchmark verified full tensor conversion, vision-projector export and serving on an L40S; v4 measured the JSON-property-order correction with the same exported model hashes.

The BF16 merge and GGUF conversion are larger than the earlier NF4 training load. Plan for **at least 100 GB host RAM and more than 200 GB of temporary disk** for the GGUF route's base cache, merged checkpoint, BF16 GGUF, and Q4 GGUF, then verify the live Pod allocation before launch. The BF16 route does not create either GGUF file, but still needs space for the base cache and merged checkpoint. The worker checks visible GPU type/count/VRAM and effective RAM/CPU against the recipe before merging, and records the measured values. A training A40 Pod's 50 GB host RAM and 140 GB workspace volume are insufficient for the GGUF export. Its example recipe requests an L40S, 120 GB host RAM and a 350 GB volume. A live L40S reported **46,068 MiB = 44.988 GiB** to `nvidia-smi`; the old 45 GiB recipe therefore failed its preflight guard before model work. The current GGUF example uses a 44 GiB minimum, while retaining the measured-memory check. Its L40S results cannot establish fit or latency on the 24 GB L4 currently used by GCP inference.

Preparation uses the original held-out **validation** cohort, with all renderings of one underlying task kept in its original split. It copies the selected export's processor, compact prompt, closed vocabulary, schema and chat template. The final assessment cohort stays reserved. Benchmark reports preserve raw output and explicit predictions alongside deterministic canonicalization and ontology validation; derived closure is not scored as an explicit model prediction. The primary result is exact-set match, supported by precision/recall/F1, per-dimension, per-view, frequency slices, invalid-output counts, startup time, peak GPU memory and cost estimates. On the GGUF route, per-example `latency_seconds` includes image hashing, RGB JPEG conversion, request construction and the server response; `server_latency_seconds` isolates the HTTP/server roundtrip. On the BF16 route, the per-example timer starts at processor batch creation after image rendering and includes generation and decoding. Throughput covers each route's timed round, so the different preparation boundaries matter when comparing speeds. Runpod latency and startup measurements describe the chosen Runpod GPU and storage path; they do not establish Cloud Run cold-start behavior.

The GGUF recipe pins llama.cpp's `ubatch_size` at 1024. The pinned server otherwise defaults to 512 and caps the Qwen vision image-token budget to that size; four of the 64 validation images had more than 512 image tokens in the training renderer (maximum 696). An explicit 1024-token microbatch avoids that known downscaling while the GPU benchmark verifies its memory fit.

## Merged BF16 diagnostic

Start from [the BF16 example recipe](../experiments/inference-benchmark-qwen38-27b-bf16-v1.json), assign a new run ID and use the same `inference-benchmark check-config`, `prepare`, `stage`, `render`, `launch`, `status` and `stop` commands below with that recipe and a reviewed image digest. Its `benchmark.route` is `merged_bf16_hf`; it requires concurrency one, the original 512-token output ceiling and `expected_merged_files_digest`. That digest is SHA-256 of the completed GGUF export's `merged_hf.files_sha256` map serialized with `json.dumps(..., sort_keys=True, separators=(",", ":"))` and UTF-8 encoding. It has no GGUF quantization, microbatch or llama.cpp revision setting.

This is a focused quality check of the checkpoint before GGUF conversion. It uses the same prepared 64 validation rows and their recorded image, prompt, processor, schema, ontology and render hashes. After merging the adapter into the pinned BF16 base, the worker verifies the merged file hashes against that expected digest. It then uses the training renderer and Transformers constrained-generation path with greedy decoding and the original 512-token ceiling. Each prediction and the resulting evaluation are retained in a report; this route does not publish a model candidate or change production inference.

Full-GPU loading requires more memory than the measured L40S's 44.988 GiB because the merged checkpoint alone is roughly 54 GB. Review an 80 GB-class GPU's live VRAM, host RAM, disk and price before launching this diagnostic. The route uses full-GPU loading so it fails instead of silently offloading weights to the CPU. A concurrency-one replay is sufficient for the primary paired accuracy check; the GGUF route's concurrency-two and concurrency-four rounds serve a separate capacity question.

The first A100 BF16 attempt completed merging and matched the GGUF export's merged-file digest, but PyTorch's native Triton build path failed on the first generation warmup before any prediction. The corrected v2 request included `TORCH_DISABLE_NATIVE_JIT=1`, as the successful training request did, and completed both concurrency-one repeats at **40/64 exact-set match (62.50%)**, **95.23% label F1** and zero invalid outputs. The two repeats produced identical generated text. The [run record](RUNPOD-BF16-BENCHMARK-20261005.md) preserves the failure, corrected replay, contract and report references. A further paid GPU run still needs a new run ID and explicit user approval.

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

For GGUF, start from [the current example recipe](../experiments/inference-benchmark-qwen38-27b-gguf-v3.json); for merged BF16, use the recipe linked above. Review it and assign a new run ID for a new benchmark; substitute the copied recipe path in the commands below. Local preparation needs the original prepared training manifest and a downloaded, verified selected model bundle. The source bundle is an immutable GCS object recorded in the [full-run record](RUNPOD-FULL-A40-20261002.md); its exact URI, size and SHA-256 go in the recipe. Obtain the bundle through the approved read-only recovery path and keep it in `artifacts/` or another gitignored artifact directory. Do not put model weights, credentials or generated JSONL in the repository root.

```powershell
uv sync --group training --group inference-benchmark
uv run --no-sync edugraph-classify inference-benchmark check-config --config 'experiments/inference-benchmark-qwen38-27b-gguf-v3.json'
uv run --no-sync edugraph-classify inference-benchmark prepare `
  --config 'experiments/inference-benchmark-qwen38-27b-gguf-v3.json' `
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

`stage` records `runs/<benchmark-run-id>/staged.json`; `render` writes `pod-request.json`; `launch` writes `pod-launch.json` before contacting Runpod so an ambiguous response cannot silently create a duplicate Pod. The worker checks its baked code commit against the manifest and also checks the llama.cpp commit on the GGUF route. It verifies staged inputs and source hashes, then writes a completion marker under the configured benchmark artifact prefix. Failures are printed with sanitized exception locations before the worker stops its Pod. After GCS access is available, a preflight failure writes `attempts/<run-id>/<pod-id>/failure.json`; once the prepared manifest is verified, that marker points to an immutable failure bundle that can also contain completed-round progress. Server logs are omitted from failure bundles because they may contain input content. A successful benchmark terminates the Pod after durable publication; an interrupted or failed run needs status and billing reconciliation before another launch. If the stop or termination API fails, the worker emits a sanitized `shutdown_failed` event, and the Pod must be checked manually. Follow [Runpod training](RUNPOD-TRAINING.md) for the existing Secure Cloud, credential and launch boundaries.

## Interpreting the result

Compare each configuration with the selected NF4-plus-adapter report on the same 64 validation IDs and gold labels, using the exact-set evaluator and paired per-case outcomes. The offline `inference-benchmark compare` command in the [model-to-production workflow](MODEL-TO-PRODUCTION.md) makes this replay reusable across the raw training, BF16, and GGUF reports. It cross-checks the selected source export's manifest, result and controls against the original training manifest; compares each benchmark's copied control and image hashes with that original; replays reported metrics; and verifies matching merged-model hashes. It writes paired regressions and recoveries under `reports/` or `temp/`, including all concurrency-one BF16/GGUF repeat combinations when both routes are supplied. Verify the source bundle against the immutable training receipt and downloaded report archives against their completion-marker SHA-256 first. The measured exact-set results are **43/64 for NF4 plus adapter, 40/64 for merged BF16 Hugging Face, and 37/64 for corrected Q4_K_M GGUF**. The merged BF16 model uses the training rendering and constrained-generation implementation, but NF4-to-BF16 also changes the base weight precision; the three-case difference cannot be assigned to adapter merging alone. BF16 Hugging Face to Q4_K_M GGUF additionally changes conversion, quantization, serving implementation and JSON constraint mechanics. Its three-case aggregate difference is therefore not a measurement of quantization loss alone. A BF16 GGUF replay through the same llama.cpp path would isolate the quantization step more closely if that distinction becomes necessary. The v3 run exposed a JSON-property-order mismatch in the llama.cpp generation path. The request builder now sends a copy of the same closed schema with `areas, scopes, abilities` in the training order; a controlled v4 replay with matching model hashes improved exact-set match from 19/64 to 37/64. Compare latency, startup-to-ready time, memory peak, invalid outputs and cost only with their full hardware and runtime settings. The [GGUF run record](RUNPOD-GGUF-BENCHMARK-20261004.md) and [BF16 run record](RUNPOD-BF16-BENCHMARK-20261005.md) preserve the measured results and artifact references. The current release subsequently passed a separate [Cloud Run load and application canary](IMAGINE-BASELINE-PROMOTION.md); a future benchmark alone must never change production traffic.
