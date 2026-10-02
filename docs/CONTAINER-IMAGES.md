# Training container images

The [Runpod Secure workflow](RUNPOD-TRAINING.md) uses this image with the entrypoint `python -m edugraph_classify.runpod_worker`. The historical Vertex entrypoint remains the image default. W&B and the constrained decoder are locked in the `training` group. `CODE_COMMIT` is embedded at build time and checked alongside the lockfile before training; the publishing workflow passes the checked-out Git SHA. Local builds with `uncommitted` are verification images only.

## Registry decision

As of 2026-10-02, **GitHub Container Registry (GHCR)** is the default registry for new self-hosted training images, including the planned Runpod path. The canonical image is `ghcr.io/christian-bick/edugraph-classify-trainer`. Container images hold the code and its locked Python/CUDA environment. GCS remains the artifact store for inputs, checkpoints, predictions, and run records; the registry change does not move those objects.

GitHub currently provides free container image storage and bandwidth for both private and public packages. This repository is public and uses a standard GitHub-hosted Ubuntu runner, which is free for public repositories. Private forks and larger runners have separate Actions billing. Publication requires a manual dispatch or an explicitly pushed `trainer-*` tag; ordinary branch pushes and pull requests do not publish. No personal access token is required for publication through that workflow: it uses the job's short-lived `GITHUB_TOKEN` with `packages: write`. See [GitHub Packages billing](https://docs.github.com/en/billing/concepts/product-billing/github-packages), [Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions), and [publishing Docker images](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images).

The intended image visibility is **public**, matching the user's repository and image release decision. New GHCR packages still start private: after the first publication, set the package visibility to public separately and verify an anonymous pull of its digest. This workflow does not change visibility. Private forks require a registry pull credential on the Docker host or Runpod template; keep it in the provider's registry credential store, separate from experiment manifests and container environment variables. See [GHCR access](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

## Build and publish

The existing `containers/vertex-trainer/Dockerfile` remains the single image recipe. Its directory name reflects its origin; the image packages both the Vertex and Runpod trainers. The Runpod executor, constrained generation decoder and language-only training policy are implemented; GPU acceptance and throughput still require the smoke described in [Runpod training](RUNPOD-TRAINING.md).

The Docker context includes only `pyproject.toml`, `uv.lock`, `README.md`, `LICENSE`, `src/`, and the Dockerfile. Datasets, model weights, checkpoints, local environment files, and Git credentials are excluded. The build installs from the frozen lockfile, imports the training stack without network access, verifies the CUDA 12.6 build of PyTorch, and checks the trainer CLI. These are CPU checks; they do not establish successful GPU training.

The project is licensed under Apache 2.0. The image includes `LICENSE`, and the publishing workflow records `org.opencontainers.image.licenses=Apache-2.0`. Bundled dependencies and the CUDA base image retain their respective licenses.

To publish before the workflow reaches the default branch, explicitly push a `trainer-*` tag pointing to a reviewed clean commit. Only that tag pattern triggers publication; ordinary branch pushes and pull requests do not. This permits publishing a pinned smoke image without merging unvalidated GPU code into main. The tag authorizes image publication, not paid training.

After the workflow is available on the default branch, the manual UI is also available:

1. Open **Actions → Publish training image → Run workflow** and select the reviewed branch or tag.
2. Leave **publish** unchecked for a build-only check, or check it to build and publish. A manual dispatch authorizes the selected build and any applicable Actions charges; it does not launch training.
3. Read the workflow summary for `ghcr.io/christian-bick/edugraph-classify-trainer@sha256:<digest>`, the full code commit, and lockfile hash. The workflow also publishes the `sha-<full-code-commit>` tag and records source/revision OCI labels and build provenance.
4. Record the **digest URI** in the run manifest. Tags locate builds; only the digest fixes the actual image bytes.
5. For a private package, configure pull access before using it on Runpod. Public visibility requires a separate, intentional package setting change; publishing from a public repository does not automatically make a new package public.

All third-party workflow actions are pinned to official release commits. The workflow builds only `linux/amd64`, matching the current CUDA/GPU target. It does not upload a dataset or contact a training provider. Build records are not uploaded as separate Actions artifacts.

To build and inspect locally without publication or cloud build charges, run from the repository root:

```bash
docker build --platform linux/amd64 \
  -f containers/vertex-trainer/Dockerfile \
  -t edugraph-classify-trainer:local .
docker run --rm --network none edugraph-classify-trainer:local --help
```

## Existing images and Vertex compatibility

Historical experiment configurations retain their original image repositories and digests as provenance. On 2026-10-02, the user-authorized [GCP cleanup](GCP-CLEANUP.md) deleted both `training` repositories and the old Qwen3-VL 4B/8B trainer repositories. Those historical image URIs no longer resolve. Reproducing an old run now requires a separately retained image or a rebuild from its matching source and lockfile; a rebuild is not guaranteed to reproduce the original digest.

**Production inference remains on GCP with scale-to-zero.** Its `edugraph-predict` repository and earlier `gcr.io/llama-server` revision images remain, along with the separate Imagine application's repositories. GHCR is the destination for new self-hosted training images; the cleanup did not change Cloud Run deployment images, traffic, scaling, GCS storage, or access controls. The Artifact Registry API remains enabled for those retained consumers.

Google's custom-training documentation currently lists **Artifact Registry and Docker Hub** as supported registries, not GHCR. A future Vertex job therefore needs a compatible mirror and its resolved digest; do not mechanically replace the image URI in an old Vertex job with a GHCR URI. The old Vertex instructions remain historical and provider-specific. See [Google custom-container requirements](https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-container).

Fireworks serverless training uses a provider-managed runtime and does not pull this container. Runpod and local Docker are the consumers of the new GHCR publishing path.
