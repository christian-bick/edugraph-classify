# Prepare the Imagine GGUF release for Hugging Face

The selected repository name is `christian-bick/EduGraph-Classifier-Qwen3.8-27B-GGUF`. The v4 L40S candidate is still a benchmark artifact in GCS; neither that repository nor a GCP deployment is created by this workflow. The [benchmark record](RUNPOD-GGUF-BENCHMARK-20261004.md) explains the 37/64 GGUF validation result, its comparison with merged BF16 and NF4, and the request-order correction required to reproduce it.

`experiments/hf-gguf-release-qwen38-27b-v1.json` pins the v4 candidate archive's full SHA-256 and byte length, the six source-file hashes, the source training and model revisions, and the evaluation report. The local `hf-gguf-release` command checks the archive and every member before creating a release directory. It rejects extra, duplicate, non-file, or renamed tar members; it never extracts arbitrary paths. The release contains only `model-Q4_K_M.gguf`, `mmproj-BF16.gguf`, `chat_template.jinja`, `prompt.json`, and `closed_schema.json`, plus a generated model card and `provenance.json`. Operational `manifest.json` and `candidate.json` are verified but deliberately omitted from the public staging area. The source GCS URI, Pod and local paths are not copied into the card or provenance.

Use a locally available, verified copy of the candidate archive. Obtaining the 17.5 GB source archive from GCS and uploading a Hub repository are separate provider actions, outside this command. Run from the repository checkout:

```powershell
uv run --no-sync edugraph-classify hf-gguf-release `
  --recipe experiments/hf-gguf-release-qwen38-27b-v1.json `
  --candidate-tar <path-to-v4-candidate.tar> `
  --output artifacts/hf-gguf-release-qwen38-27b-v1
```

The staged card initially says **DRAFT — DO NOT PUBLISH**, has no Hub license metadata, and includes `DO_NOT_PUBLISH_LICENSE_PENDING.txt`. This is intentional: the repository's Apache-2.0 **code** license does not decide the license for the fine-tuned **weights**. After the model owner chooses the weight license and provides its complete terms as a local file, repeat into a new output directory with both `--model-license-id <hub-license-id>` and `--model-license-file <path-to-approved-license>`. The command copies that file to `LICENSE`, sets the card's Hub license metadata and records the chosen ID in provenance. Supplying only one value fails. A passing gate verifies that an explicit choice was supplied; it does not substitute for legal or release review.

Review the complete staged directory before any upload. In particular, confirm the two GGUF file hashes, that the card describes the **same 64-image original validation cohort** (GGUF 37/64, merged BF16 40/64, selected NF4-plus-adapter 43/64), and that the separate **181/250 final assessment belongs only to NF4 plus adapter**. Check the `llama-server` command and request contract in the card, including JPEG quality 75, a one-image user message, greedy generation, JSON response format, and request-time `areas, scopes, abilities` schema property order. The latter fixed a large v3-to-v4 accuracy difference without changing the GGUF weights.

The command is preparation only: it has no Hugging Face SDK, authentication, repository-creation, or upload path. Publishing the chosen weights, and any GCP promotion, require their own reviewed steps. A Hub release should use these exact staged files and preserve the card, license, and provenance together. Do not substitute the 4B legacy model card or report its score as this GGUF model's result.
