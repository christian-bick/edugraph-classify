# Runpod Secure training with W&B

Implemented 2026-10-02. The bounded **Secure A40 smoke completed** with authenticated W&B and GCS access, 40 optimizer steps, generated before/after evaluation, verified external checkpoints and automatic Pod termination. Held-out label F1 improved from 3.18% to 61.25%; exact-set match stayed 0/64. The live audit confirmed language-only adapters and zero trainable vision/bridge parameters. The user-authorized [full two-epoch A40 run](RUNPOD-FULL-A40-20261002.md) launched at 21:27 UTC with overnight monitoring; results are pending. See the [smoke record](RUNPOD-SMOKE-20261002.md) for its pins and measured results. Production inference remains on GCP. The Fireworks executor and experiment records remain available for comparison.

## Pinned training policy

| Setting | Full candidate |
|---|---|
| Recipe | `experiments/edugraph-20261002-qwen38-27b-runpod-full-a40-v1.json` |
| Model | Qwen/Qwen3.8-27B, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Dataset / ontology | v0.30.0-03 / 0.30.0, same immutable pins as the compact Fireworks recipe |
| Executor | One on-demand Runpod Secure Pod, NVIDIA A40 48 GB |
| Host allowance | At least 48 GB RAM and 8 vCPUs; 30 GB container disk and 140 GB `/workspace` volume |
| Training | NF4 QLoRA, double quantization, BF16 compute, rank 8 / alpha 16, dropout 0 |
| Trainable modules | Audited language attention and MLP adapters; vision tower, multimodal bridge, embeddings and output head frozen |
| Batch / optimizer | Microbatch 1, effective batch 4; token-weighted assistant-only loss; constant AdamW LR 1e-4, gradient norm limit 1 |
| Context / response | 2,048 expanded tokens / at most 512 generated tokens; no truncation |
| Epochs | Two total, patience-one early stopping based on exact-set match then label F1 |
| Checkpoints | Every 50 optimizer steps and each epoch, plus before final assessment |
| W&B | Entity `edugraph-io`, project `edugraph-classify` |
| Durable artifacts | `gs://edugraph-classify/runpod` |

The separate `edugraph-20261002-qwen38-27b-runpod-smoke-v2.json` recipe uses 160 train images, 64 validation images, 16 train diagnostics, one epoch, and checkpoints every 10 steps. It leaves the final assessment cohort unused. On 2026-10-02 the live Secure inventory had no A6000 allocation, but offered an **A40 with 48 GB VRAM, 50 GB host RAM and 9 vCPUs at $0.49/hour**. The smoke and authorized full run explicitly select A40 and request at least 48 GB host RAM / 8 vCPUs. Transformers loads quantized shards directly to the GPU; this does not require a full BF16 CPU model copy. The smoke measured **29.122 GiB peak PyTorch device allocation**; the later L40S benchmark passed the longest training inputs. The original `runpod-v1` A6000 recipe remains available as an earlier candidate; the committed `runpod-full-a40-v1` recipe is the actual full-run configuration. The legacy recipe key `minimum_ram_gib` is passed to Runpod's GB-based `minRAMPerGPU` filter.

**Secure Cloud is mandatory**, with no Community or ALL fallback. Both the launcher and worker require `machine.secureCloud == true` from the live Pod response. A missing or conflicting flag stops the Pod. This is the selected provider trust boundary, not confidential computing against the operator. The user explicitly authorized the full two-epoch A40 run and overnight monitoring on October 2 after the smoke and hardware benchmarks.

The A40 recipe sets `minimum_vram_gib: 45` for usable device memory, with the loader's existing 1 GiB tolerance. This still selects the same nominal 48 GB A40. Advertised capacity includes driver/ECC reservations, so the original nearly-48-GiB check could reject some intended allocations. No device ECC setting, example or training policy was changed. This specific Pod subsequently reported **47.404 GiB usable**, which also exceeds the original threshold. See [NVIDIA's memory-capacity note](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a40/NVIDIA%20A40%20Product%20Brief.pdf). The actual launch used this recipe from an ignored operational file to preserve its clean image/code pin; the manifest and staged bundle record the complete configuration.

The same compact v3 **system message** and user message feed training and validation. Ontology vocabulary is supplied through `closed_schema.json`, rather than added to the prompt. `chat_template.jinja` embeds the system message for later inference integration. Every validation phase generates predictions; loss is diagnostic only. Label ordering does not affect exact-set match or F1; targets retain deterministic alphabetical serialization. Explicit and canonicalized scores, invalid outputs, per-view/per-dimension/frequency slices, raw responses, latency and token usage remain in GCS reports.

The self-hosted baseline is the frozen **NF4-quantized** model; it is not assumed identical to Fireworks' managed precision. The Fireworks adapter/state is not imported into QLoRA. This candidate starts from the pinned public base model.

## Credentials and tracking

The screenshots identify W&B organization `edugraph-io-org`, team/entity `edugraph-io`, service account `edugraph-classify`, and service-account key named `edugraph-classify-runpod`. The user's Runpod Secret is named `WANDB_API_KEY`. The request maps:

```text
WANDB_API_KEY = {{ RUNPOD_SECRET_WANDB_API_KEY }}
EDUGRAPH_GCS_CREDENTIALS_JSON = {{ RUNPOD_SECRET_edugraph-gcs }}
```

The W&B **key value**, rather than its displayed key ID, belongs in the secret. The SDK in the Pod authenticates independently of MCP. No W&B MCP tools were exposed to this session; MCP is optional for querying runs and is not needed to train or log metrics. The v2 Secure smoke successfully authenticated and created its W&B run on 2026-10-02. Its GCS credential also downloaded the prepared bundle and wrote the worker startup marker.

The GCS secret `edugraph-gcs` is configured for `edugraph-runpod-training@edugraph-438718.iam.gserviceaccount.com`. It has conditional `roles/storage.objectViewer` and `roles/storage.objectCreator` bindings restricted to `gs://edugraph-classify/runpod/`. The existing bucket uses a hierarchical namespace, so the condition covers both `objects/runpod/` and `folders/runpod/`; the latter permits creating the training subfolders. It grants no object deletion, project administration, compute, or production-serving access. A live read/create probe passed and a read outside the prefix returned 403. The JSON key was transferred directly from Google to Runpod Secrets in memory, without writing its value to disk or logs. Failed setup keys were revoked. Do not copy personal ADC into a Pod. `GcsBundles` also supports standard ADC when the secret is absent, for example a separately configured workload identity mounted into the container; provisioning federation is outside this recipe.

The workstation uses `RUNPOD_API_KEY` with REST v1 to create/read/stop Pods and local Google ADC to stage inputs. Keep that account key in ignored `.env` or the environment. **It is not injected into the Pod.** Runpod supplies a Pod-scoped `RUNPOD_API_KEY` and `RUNPOD_POD_ID`; the worker uses **GraphQL** to inspect, stop and terminate its own Pod. Live testing on 2026-10-02 found that the same Pod-scoped key receives 403 on REST v1 but works on GraphQL and REST v2. GraphQL get/stop/terminate were all verified on the first failed smoke Pod before its removal. Public model/data downloads and the public GHCR image require no Hugging Face or registry token. See [Runpod secrets](https://docs.runpod.io/pods/templates/secrets) and [Pod environment variables](https://docs.runpod.io/pods/templates/environment-variables).

W&B receives pinned configuration, losses, gradient norms, generated-validation metrics, GPU memory, timings and session cost estimates. Adapter weights and optimizer states stay in **GCS**; each W&B artifact contains a small downloadable `gcs-reference.json` with its `gs://` URI, SHA-256, run ID and size. This file can be passed directly to `--resume-reference`. The pinned W&B SDK initializes Google ADC even for native GCS references with `checksum=False`; using reference JSON avoids a second Google credential path. A real-SDK offline test verifies that no ADC is requested. No `wandb.watch`, automatic code upload, console capture or duplicate tensor uploads are enabled. Session cost starts at the trainer and excludes earlier provisioning/input staging, stopped storage, GCS charges and unsaved work lost to an outage; it is not an invoice.

## Local preparation and explicit launch

```powershell
uv sync --group training --group training-api
uv run --no-sync edugraph-classify runpod check-config
uv run --no-sync pytest --cov=edugraph_classify --cov-report=term-missing
```

Commit reviewed code and lockfile, then publish the image from that commit through the [GHCR workflow](CONTAINER-IMAGES.md). Use its **digest**, not a mutable tag. The image embeds `EDUGRAPH_CODE_COMMIT`; the worker checks this and `/app/uv.lock` against preparation. Local builds with `CODE_COMMIT=uncommitted` are verification images and cannot launch pinned experiments.

Prepare the smoke from a clean checkout; this downloads public inputs locally without creating provider resources:

```powershell
uv run --group training edugraph-classify runpod prepare `
  --config experiments/edugraph-20261002-qwen38-27b-runpod-smoke-v2.json
```

The following operations are separate and require explicit user authorization. `stage` uploads to GCS; `launch` creates a paid Pod. Both require exact run-ID confirmation. The examples below reproduce the earlier smoke; the separately authorized full A40 run and its reviewed request are recorded in [the full-run log](RUNPOD-FULL-A40-20261002.md).

```powershell
$runId = 'edugraph-20261002-qwen38-27b-runpod-smoke-v2'
$manifest = "runs/$runId/manifest.json"
$image = 'ghcr.io/christian-bick/edugraph-classify-trainer@sha256:<published-digest>'
uv run --no-sync edugraph-classify runpod stage --manifest $manifest --confirm-run-id $runId
uv run --no-sync edugraph-classify runpod render --manifest $manifest --image $image
uv run --no-sync edugraph-classify runpod launch --manifest $manifest --image $image --confirm-run-id $runId
uv run --no-sync edugraph-classify runpod status --launch-record "runs/$runId/pod-launch.json"
```

`pod-request.json` is reviewable and contains secret references only. An exclusive launch journal is written **before** calling Runpod. An ambiguous network error may mean a Pod exists; inspect Runpod before retrying. Do not remove the journal to work around uncertainty. No inference endpoint, public ports or SSH service is provisioned.

The GPU ceiling is **$0.60/hour**, checked after creation and again by the worker before model loading. Numeric and string currency values are accepted; missing, nonfinite and excessive rates fail closed. The documented REST create request has no price-cap field, so a rejected allocation can incur brief rental time before stopping. The authorized full A40 recipe allows 8 hours from worker startup; the smoke allows 4 hours. A watchdog calls stop at the deadline. This is not a guaranteed billing cap: image-pull/startup failures precede the worker, and an outage can defeat its stop request. Monitor startup and verify shutdown. See the [REST contract](https://docs.runpod.io/api-reference/pods/POST/pods).

Success publishes the selected model and completion marker, flushes W&B, then **terminates the Pod**, removing its volume. Failure or a cooperative pause stops the Pod while preserving its billable volume. An immediate `runpod stop --launch-record ... --confirm-run-id ...` releases compute; recover from the last external checkpoint. Do not assume a Pod disk survives host loss or that a stopped Pod can reacquire its GPU.

## Recovery and one-epoch continuation

Checkpoints contain the current adapter, AdamW state, Python/Torch/CUDA RNG states, data position, global step, best adapter, selection history, reports and source manifest. The best adapter is separate from the current training adapter so optimizer state remains coherent. Constant LR needs no scheduler state. CPU tests compare uninterrupted and interrupted/resumed updates and image order; cross-GPU bitwise identity is not promised.

To recover on a fresh Pod, save the desired W&B/GCS checkpoint reference locally with `run_id`, `uri`, and `sha256`. The worker also writes `/workspace/training/latest-checkpoint.json`. Use the original prepared run and an unused attempt name:

```powershell
uv run --no-sync edugraph-classify runpod launch `
  --manifest $manifest --image $image --confirm-run-id $runId `
  --resume-reference reports/recovery-checkpoint.json --attempt recovery-1
```

Launch downloads and verifies compatibility **before** creating the replacement. Same-run recovery resumes its W&B ID and uses a separate `pod-launch-recovery-1.json` journal. A running/ambiguous Pod for the run blocks another allocation; stop and reconcile it first. Updates after the last checkpoint are replayed.

For **one epoch now and more later**, start with `training.epochs: 1`. Then create a new recipe with a new run ID and a larger **total** epoch target (2 means one additional epoch after epoch 1). Prepare, stage and launch it with the continuation reference. A run-specific recipe can live in ignored `temp/continuation.json`, preserving the original clean code commit. Resume rejects changes to code/lock/runtime, model, dataset/cohorts, prompt/schema/processor, optimizer or LoRA policy. Within training policy, only total epoch ceiling and patience may change; infrastructure/tracking can differ. Code changes require an explicitly designed checkpoint migration.

A failed final assessment can recover the same run from its `phase=final` checkpoint without repeating optimizer steps. The final export contains the selected adapter, processor, prompt, schema, template and reports. It does **not** merge, register or deploy a serving model. Completed-run markers are immutable; training continuation gets a new run ID.

## Validation and remaining acceptance

The first Secure smoke (`...runpod-smoke-v1`, Pod `yhbqhq55hz4qkx`, code `a69514d`) reached container startup but failed on the REST v1 Pod-key check, before GCS worker staging, W&B or model loading. The workstation stopped it, saved its failure record in GCS and terminated its empty volume. No optimizer steps ran. The v2 smoke completed with the corrected GraphQL worker adapter and verified automatic termination. Failure reporting preserves both the original exception and any shutdown failure as sanitized type/status/frame records.

Tests exercise provider-error sanitization, replay/price guards, secret references, immutable bundles/path safety, interrupted recovery, optimizer restoration, frozen-vision auditing, assistant masks, image expansion, constrained JSON and generated checkpoint selection. The actual cached Qwen tokenizer (248,077 tokens) accepted closed-schema JSON; the largest audited image expanded to 1,147 image tokens with matching multimodal type IDs. The meta-model audit found 496 selected language linear modules and zero vision/bridge targets. Local evidence is under ignored `reports/runpod-setup-20261002/`.

The worker also emits explicit progress events for loading, generated predictions, optimizer steps and durable checkpoints. Remote failure reports include exception types and frame locations, excluding exception values and local variables. No console output is forwarded automatically to W&B.

The packaged lm-format-enforcer Transformers integration imports a removed symbol. Our small adapter uses its core `TokenEnforcer` API and preserves the exact Qwen tokenizer. Constraints enforce syntax and vocabulary; output truncation or decoder failure can still produce invalid output, and semantics still require evaluation.

The smoke established enough learning evidence to stop after one epoch. The subsequent [Secure L40S benchmark](RUNPOD-HARDWARE-BENCHMARK.md) restored its external checkpoint, verified all 992 optimizer states at step 40, executed the next real batch to step 41 and published coherent continuation state. The longest 1,281/1,302-token training inputs also passed; peak tensor allocation was 29.949 GiB, with 37.051 GiB device memory in use at probe completion. Retain a nominal 48 GB GPU. The longest-prompt generation ended after 30 tokens, so the full 512-token output ceiling was not forced.

The decoder left duplicate labels in 8/64 held-out smoke outputs; canonicalization repaired validity without changing their set-based scores. Exact-set match remained zero even after canonicalization, so this is not a production-quality checkpoint. The measured A40 planning range is 5–6 hours and $2.57–$3.08 for two full epochs; L40S is 4–5 hours and $4.46–$5.57 including running disk. See [training costs](TRAINING-COSTS.md) for the paired forecast and uncertainty. The full A40 run was explicitly authorized and launched after a fresh capacity/rate check; its generated assessment is pending. Immutable checkpoints are retained; review retention after recovery validation rather than deleting recovery data automatically.
