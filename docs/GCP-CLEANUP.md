# GCP cleanup record

On **2026-10-02**, the user authorized removal of obsolete Docker registry storage in project `edugraph-438718`, while retaining **GCS model artifacts** and **production inference on GCP with scale-to-zero**. Four unused training repositories were deleted. A subsequent explicit request authorized removal of the retired Hermes VM, its disks, and dedicated leftovers; that cleanup is also complete. Production services and their deployment/rollback images remain available.

## Removed training repositories

| Repository | Location | Image records | Stored GiB | Estimated USD/month removed |
| --- | --- | ---: | ---: | ---: |
| `training` | `europe-west3` | 1 | 5.61 | $0.56 |
| `training` | `europe-west4` | 8 | 22.16 | $2.22 |
| `edugraph-qwen-3vl-8b` | `europe-west4` | 3 | 10.18 | $1.02 |
| `edugraph-qwen-3vl-4b` | `europe-west4` | 194 | 126.70 | $12.67 |
| **Total** | | **206** | **164.65** | **$16.46** |

Sizes use the API's repository-level `sizeBytes`, not the sum of individual image sizes, because image layers can be shared. Unrounded removed storage was 176,788,853,749 bytes. The estimate uses Google's current storage rate of $0.000136986/GiB-hour, approximately $0.10/GiB-month over 730 hours. Remaining storage still exceeds the billing account's 0.5 GiB free tier, so the estimated marginal saving is $16.46/month. Totals use unrounded values. These are storage estimates, not invoice totals; currency conversion, taxes, transfer, and other services are separate. See [Artifact Registry pricing](https://cloud.google.com/artifact-registry/pricing).

Before deletion, live dependency checks found no training-image references in any of the 43 retained Cloud Run revisions. All 12 Vertex custom jobs were terminal failures, and no Vertex endpoints, persistent training resources, or Cloud Run jobs were found in `europe-west3` or `europe-west4`. Repository metadata, image identities, and Cloud Run service configuration were rechecked immediately before deletion.

All four deletion operations completed successfully by **12:42:58 UTC**. A subsequent GET for each repository returned HTTP 404. Historical run manifests and source configurations retain their original image references as provenance, but those deleted image digests can no longer be pulled. Reproducing a historical run requires a separately retained image or rebuilding its matching source and environment; rebuilding does not guarantee an identical digest.

## Retained production and storage resources

| Repository | Location | Stored GiB | Reason retained |
| --- | --- | ---: | --- |
| `edugraph-predict` | `europe-west4` | 4.64 | Current production inference and revisions |
| `gcr.io` (`llama-server`) | `us` | 4.46 | Earlier production inference revisions |
| `imagine-server` | `europe-west4` | 21.78 | Separate running application and revision history |
| `firebaseapphosting-images` | `europe-west4` | 0.10 | Separate Imagine application build images; not part of the training replacement |

Remaining registry storage is **30.98 GiB**, approximately **$3.05–$3.10/month** depending on whether the shared free allowance is available. Removing obsolete training images reduced inventoried registry storage by about **84%**. The small retained production registry cost is independent of training's move to GHCR.

The `edugraph-predict` Cloud Run service retains revision `edugraph-predict-00005-59s`, 100% of traffic, an L4 GPU, and its original image digest `sha256:6c0080fc4ff1ddda4295916a44ec6cc8e715ab3d01d3b5172b7def6c5cdfb9f0`. Both service and revision minimum instance settings are omitted (the default is zero); the maximum is one. Scale-to-zero remains configured. This was an administrative configuration check; no inference request was sent and no deployment was created or scaled. See [Cloud Run minimum instance defaults](https://docs.cloud.google.com/run/docs/configuring/min-instances).

Before/after service projections matched for both `edugraph-predict` and `imagine-server`, including image references, traffic, and scaling. All four protected repositories still returned HTTP 200. All six GCS buckets were retained, with no object writes/deletions or IAM changes:

- `edugraph-classify`
- `edugraph-classify-europe-west3`
- `edugraph-embed`
- `imagine-ml`
- `imagine-content`
- `edugraph-438718_cloudbuild`

Runtime identities, storage permissions, and enabled APIs were retained. In particular, production still uses Artifact Registry, so its API must remain available. GHCR publication is a separate workflow and was not performed by this cleanup.

## Hermes retirement

Following the user's explicit request, these resources in `europe-west3-c` were deleted on **2026-10-02**, with verification completed at **12:59:18 UTC**:

| Resource | Result |
| --- | --- |
| VM `hermes-agent-prod` | Deleted from its stopped state; no restart or compute launch |
| 50 GiB standard boot disk `hermes-agent-prod` | Automatically deleted with the VM under its existing auto-delete setting |
| 100 GiB balanced disk `hermes-persistent-data` | Deleted after the VM released it |

All three resources returned HTTP 404 after completion. The final aggregated Compute inventories contained **zero VMs and zero persistent disks**. No snapshot or replacement storage was created, so the 150 GiB of allocated disk storage has been removed rather than transferred to another billable resource.

The dependency audit found no snapshots, custom images, machine images, reserved addresses, resource policies, reservations, instance templates, managed instance groups, monitoring alert policies, or uptime checks. Cloud Asset search found only the three Hermes resources above. The VM used the shared default network, subnet, and Compute service account; these and all four shared default firewall rules remain. Cloud Scheduler's API was disabled, preventing direct scheduler inspection; no Hermes schedule was found in the available asset inventory.

Before/after checks confirmed unchanged Cloud Run image/traffic/scaling configuration, GCS bucket names, and shared firewall rules. No GCS objects, service-account credentials, IAM bindings, or application container images were changed. Evidence is saved in `reports/gcp-cleanup-20261002/hermes-before.json`, `hermes-cleanup-plan.json`, and `hermes-cleanup-result.json`; the scripts are in `temp/gcp-cleanup/`.

## Imagine release cleanup on 2026-10-07

The Imagine backend's original 4B revision `imagine-server-00038-hjn` had a
revision-level minimum of one instance and instance-based CPU billing. A
replacement revision, `imagine-server-legacy-zero-20261007`, uses the **same
pinned image and legacy request contract**, but sets the minimum to zero,
maximum to one, and request-based CPU billing. A zero-traffic tag passed
`/`, `/ontology`, `/taxonomy`, and the old browser's multipart
`/classify_and_search` upload. After an ETag-guarded, validate-only-tested
traffic change, the untagged `https://api.edugraph.io` route passed the same
upload. The two uploads returned HTTP 200 with matching dimension counts and
six neighbors. The new revision is 100% of backend default traffic; the
Qwen3.8-27B backend retains its `qwen38-classify-v2` tag and 0% default
traffic. The original warm revision and its image are retained for rollback
with no tag or traffic. The 4B predictor default, 27B predictor tag, and
Firebase Hosting version were unchanged. The backend's
`docs/BLUE-GREEN-RELEASE.md` is the current rollout/rollback runbook.

After confirming no revision, tag, job, or service depended on it, the
abandoned backend revision `imagine-server-qwen38-73b09e0` and its sole
Artifact Registry index, amd64 child image, and attestation were deleted.
The child image record was 1,244,661,589 bytes; shared-layer accounting
means this is **not** a measured storage saving. Three obsolete Cloud Build
source objects totaling 12,486 bytes were also removed by exact generation.
The reviewed old rollback and current green image digests remain pullable.

The redundant failed-v3 GGUF candidate archive (17,478,901,760 bytes) was
deleted by its exact GCS generation after verifying that the successful v4
archive remained at generation `1791147881432887` with the same model and
contract hashes. The bucket's seven-day soft-delete policy may delay storage
savings. Benchmark reports and input bundles remain available for provenance.
Sanitized cleanup and legacy-smoke receipts are retained locally in
`reports/gcp-cleanup-20261007/`; the canonical release identities are in
the tracked promotion and deployment records.

## Other cost-saving candidates

These were investigated on 2026-10-02. The first two rows have the dated
follow-up above; the others remain unmodified because their application data
or operating behavior has not been cleared for removal.

| Resource | Finding | Next decision |
| --- | --- | --- |
| Cloud Run `imagine-server` | On 2026-10-02, the then-serving revision had minimum 1 and maximum 1 instance | Resolved 2026-10-07 with the tested scale-to-zero legacy replacement; original revision retained only for rollback |
| `imagine-server` registry history | On 2026-10-02, 138 image records occupied 21.78 GiB | One abandoned image chain removed 2026-10-07; retain the remaining history until rollback dependencies are checked individually |
| `imagine-ml` and `imagine-content` | Latest storage samples show 71.08 GiB and 17.85 GiB respectively | Preserve model/application data; consider lifecycle rules only for clearly identified disposable caches or temporary checkpoints |
| Old Vertex job/model/TensorBoard records | No active custom training or deployed Vertex endpoints observed in the checked regions | Retain provenance; a historical job or enabled API alone is not evidence of running GPU cost |

Persistent disk billing continues while a VM is stopped; see [Google's disk cost rules](https://docs.cloud.google.com/compute/docs/disks#cost-considerations). Cloud Run minimum instances incur charges; see [minimum-instance billing](https://docs.cloud.google.com/run/docs/configuring/min-instances#billing). The Cloud Build staging bucket was only about 0.01 GiB in the latest storage sample, so its cleanup has little savings potential.

## Evidence and coverage

Sanitized, gitignored evidence is in `reports/gcp-cleanup-20261002/`: `inventory-before.json`, `details-before.json`, `cleanup-plan.json`, and `cleanup-result.json`. Inventory scripts are in `temp/gcp-cleanup/`. No credentials or container environment values were recorded. The deletion journal contains exact resource and operation names plus post-deletion checks.

The project-wide Cloud Asset search returned 705 assets. Direct checks covered all eight repositories identified there, both Cloud Run services and all their revisions, aggregated Compute instances/disks/addresses, six buckets, regional Vertex resources, and build triggers in `global`, `europe-west3`, and `europe-west4`. There were no reserved Compute addresses in the aggregated inventory. Storage byte counts came from Monitoring's latest 2026-10-02 12:30 UTC sample, not a bucket-content audit or invoice.

The wildcard Cloud Run listing reported unreachable regions `europe-west15`, `me-central2`, `us-central2`, `us-east7`, and `us-west8`; Cloud Asset showed no related services there. Storage Transfer inspection was unavailable because that API is disabled, so a historical transfer-job asset was not treated as confirmed active. This is a project resource audit, not a billing-account-wide audit or proof that every Google service is unused.
