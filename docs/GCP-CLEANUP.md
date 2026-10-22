# GCP cleanup record

The sections below are dated records. The 2026-10-02 and 2026-10-07 statements
about active 4B routes describe their state **on those dates**, not the current
Imagine release. The [model-to-production runbook](MODEL-TO-PRODUCTION.md) and
[baseline promotion record](IMAGINE-BASELINE-PROMOTION.md) describe the 27B
classification rollout.

## Imagine 27B default-route cutover on 2026-10-22

The user approved replacing the 4B Imagine classifier, retiring its incompatible
request path, and removing 4B resources after validation. Before the switch,
`imagine-server-legacy-zero-20261007` and `edugraph-predict-00005-59s` each
received 100% of their service's default traffic. The public web bundle still
called the tagged 27B backend, which in turn called the tagged 27B predictor.
Both services used minimum zero and maximum one instance. A read-only inventory
counted 39 old backend revisions, five 4B predictor revisions, and 95,233,535,057
logical bytes in the specifically named `edugraph-qwen-3vl-4b/` GCS prefixes.
The two buckets have seven-day soft deletion; deletion is not immediate storage
savings.

The existing 27B tagged backend passed a fresh built-in upload: HTTP 200,
ontology counts 283/353/130, intentional disabled-search response, and exact
Area `Addition`, seven expected Scope labels, and Ability
`ProcedureExecution`. ETag-guarded traffic changes added `legacy-4b-rollback`
tags and moved the backend and predictor defaults to the existing 27B revisions
`imagine-server-qwen38-40135eb` and `edugraph-predict-00006-nej` respectively.
The untagged `https://api.edugraph.io` backend then passed the same upload.
The cleaned backend source `d8813a55302b5ebec63439f825876d3aceb544eb`
was merged in `imagine-server#2` at `8b39fa06efaccd4511aaa9e4c00f289addb80460`.
Its verified image uses OCI index digest
`sha256:ddfabed6fa4137bd187f963c6cfc6e898810afa8816d0e07bd82270c193c9631`
and linux/amd64 child digest
`sha256:29479fcf7b275f27b748137a34a417de1af44908435c82004e24530005abd97f`;
the release manifest is SHA-256
`defcf74dd7358dae344409fd41b1e7b3a5016fa35f15bfd14db26df9d41a40f1`.
Revision `imagine-server-27b-default-d8813a5` now receives 100% of backend
default traffic (generation 52) and calls the **untagged private predictor**.
Predictor generation 8 first sent 100% default traffic to `edugraph-predict-00006-nej`;
generation 9 retained that route while removing the temporary 4B tag.
The prior 27B canary, new backend tag, and new backend default passed both the
built-in and shared benchmark validation uploads with exact expected labels;
the benchmark image SHA-256 is
`e623a9ceeef189c458e1aef1ec1e19f1b05e17f9ec59528d878ad1092e1ca513`.
The 4B backend rollback tag was removed at backend generation 51; the 4B
predictor rollback tag was removed at predictor generation 9. Two simultaneous built-in uploads
on the untagged public API both returned HTTP 200 with exact labels, in 8.015
and 4.483 client seconds; this is a bounded concurrency smoke, not a latency
distribution or capacity benchmark. The new web build was published from exact
tested Hosting preview version `5478bb45b1ea67b5` to live release
`1792677401792000`. `https://demo.edugraph.io` now serves JavaScript asset
`/assets/index-FuVyqANV.js` (SHA-256
`69035370f33dc775ca95ceb0356990ebae608405abea3b64ce68df46d7eba7f1`)
and calls the untagged `https://api.edugraph.io` backend. Hosted Chrome uploads
of the built-in and shared validation image both returned HTTP 200 with exact
labels and no browser errors. The earlier 27B tag is no longer a dependency
of the live web bundle. The temporary preview channel was deleted after live
smoke.

After guarded dependency checks, **two confirmed 4B backend and five 4B
predictor revisions** were deleted. Their 23 allow-listed image manifests are
absent, along with 15 obsolete Hosting versions, including the October 7
migration version. The two exact `edugraph-qwen-3vl-4b/` GCS prefixes lost 179
generation-pinned objects totaling 95,233,535,057 logical bytes from live
listings. The buckets' seven-day soft-delete period still applies. An Artifact
Registry batch stopped on a dependent-manifest 404 and a GCS batch stopped on a
transient 503; both resumed only after fresh inventory. Final dry-runs returned
zero remaining allow-listed images, GCS objects, and Hosting versions. The
current 27B predictor, backend, live web version and 27B canary tags remain.

At the first cleanup, **37 historical backend revisions and 135 older image
manifests** were retained because their relationship to the 4B model had not
been established. They receive no traffic and were not part of that deletion.
The [cutover receipt](../experiments/imagine-27b-default-cutover-20261022.json)
records exact source and deployment pins. Historical 4B experiments and
migration evidence remain in Git as dated provenance, not active deployment
instructions.

### Attested backend provenance and follow-up cleanup

A 2026-10-22 source-attestation audit resolved the first cleanup's 37-revision,
135-manifest historical backend inventory. SHA-256-checked OCI indexes,
attestation manifests, in-toto blobs, and runtime subjects map **31 OCI
families / 93 manifests** to `imagine-server` source commits whose active route
used the old predictor client. Thirty of those families backed **34 inactive
backend revisions**, `imagine-server-00004-bnj` through
`imagine-server-00037-hfr`; one image family had no retained revision. The
remaining **14 families / 42 manifests** map to Gemini-route source and are
protected, along with their three retained revisions. The in-toto records
provide content-addressed source evidence, not an independently signed
attestation. These are application **client** images, not 4B model weights;
the confirmed 4B predictor and weight storage were handled in the first
cleanup. The local ignored audit file
`reports/imagine-27b-default-20261022/backend-provenance-assessment.md`
explains the source boundary and exact proposals.

All **34 source-proven 4B-client backend revisions** were deleted and verified
absent. The eight-image strict subset with an explicit old predictor endpoint
contributed **24 OCI manifests**, all verified absent. The guarded follow-up
helper then completed deletion of the other **23 attested 4B-client image
families / 69 manifests** with successful per-batch checks. Together with the
first cleanup, the confirmed removal totals are **36 backend revisions**
(2 + 34), **96 backend manifests** (3 + 24 + 69), five 4B predictor revisions,
20 predictor package manifests, 179 live GCS objects under the two exact 4B
prefixes, and 15 obsolete Hosting versions. The 27B backend and predictor
defaults, canaries, and live Hosting version were protected by the deletion
guards. The **14 Gemini families / 42 manifests**, three Gemini or transition
backend revisions, and six 27B backend manifests were outside the allow-lists
and retained. An independent final provider inventory is still pending; this
paragraph reports the guarded helper's completed deletions, not that later
cross-check. Artifact Registry manifest absence alone does not establish
storage savings because layers can be shared and accounting can lag. GCS
objects remain under seven-day soft deletion before hard deletion.

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
