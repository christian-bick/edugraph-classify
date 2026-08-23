"""Deterministic artifacts for Vertex-hosted provider-neutral training."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .dataset import file_sha256


class VertexArtifactError(ValueError):
    """Raised when a staged artifact is invalid or conflicts with cloud state."""


_GCS_ROOT = re.compile(r"^gs://([a-z0-9][a-z0-9._-]{1,220})(?:/([^\s]+))?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def split_gcs_uri(uri: str, *, require_object: bool = True) -> tuple[str, str]:
    match = _GCS_ROOT.fullmatch(uri)
    if match is None:
        raise VertexArtifactError("artifact URI must be a valid gs:// URI")
    bucket, object_name = match.group(1), match.group(2) or ""
    if require_object and not object_name:
        raise VertexArtifactError("artifact URI must name an object")
    return bucket, object_name


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise VertexArtifactError(f"prepared manifest {key} must be an object")
    return value


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise VertexArtifactError(f"prepared manifest {key} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class LocalArtifact:
    """One verified local file and its deterministic destination."""

    name: str
    path: Path
    uri: str
    sha256: str
    size: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "uri": self.uri,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class VertexStagingPlan:
    """A pure, hash-addressed upload plan and runtime manifest."""

    run_id: str
    artifacts: tuple[LocalArtifact, ...]
    runtime_manifest: bytes
    runtime_manifest_uri: str
    runtime_manifest_sha256: str
    output_uri: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "artifacts": [item.to_mapping() for item in self.artifacts],
            "runtime_manifest": {
                "uri": self.runtime_manifest_uri,
                "sha256": self.runtime_manifest_sha256,
                "size": len(self.runtime_manifest),
            },
            "output_uri": self.output_uri,
        }


class ArtifactWriter(Protocol):
    """Minimal immutable object-store boundary used by the staging workflow."""

    def put_file(self, artifact: LocalArtifact) -> None: ...

    def put_bytes(self, uri: str, content: bytes, sha256: str) -> None: ...


def _verified_artifact(name: str, path: Path, expected_sha256: object, uri: str) -> LocalArtifact:
    if not path.is_file():
        raise VertexArtifactError(f"{name} artifact does not exist: {path}")
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise VertexArtifactError(f"{name} artifact has an invalid recorded sha256")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise VertexArtifactError(f"{name} artifact does not match its recorded sha256")
    return LocalArtifact(name, path, uri, actual, path.stat().st_size)


def build_vertex_staging_plan(manifest_path: Path, bucket_uri: str) -> VertexStagingPlan:
    """Verify a prepared run and construct content-addressed Vertex inputs."""

    bucket, root = split_gcs_uri(bucket_uri, require_object=False)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("manifest_version") != 1:
        raise VertexArtifactError("prepared manifest version must be 1")
    run_id = _string(payload, "run_id")
    code_commit = _string(payload, "code_commit")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", run_id):
        raise VertexArtifactError("prepared manifest run_id is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise VertexArtifactError("prepared manifest code_commit is invalid")

    provider = _mapping(payload, "provider")
    expected_provider = {
        "provider_id",
        "training_execution_mode",
        "project_id",
        "location",
        "bucket_uri",
        "service_account",
    }
    if set(provider) != expected_provider:
        raise VertexArtifactError("Vertex provider configuration has unexpected fields")
    if provider.get("provider_id") != "gcp_vertex_ai" or provider.get(
        "training_execution_mode"
    ) != "serverless":
        raise VertexArtifactError("prepared manifest is not a Vertex serverless run")
    configured_bucket = _string(provider, "bucket_uri").rstrip("/")
    if configured_bucket != bucket_uri.rstrip("/"):
        raise VertexArtifactError("staging bucket does not match the prepared manifest")

    model = _mapping(payload, "model")
    if set(model) != {"repository", "revision", "created_at", "capabilities"}:
        raise VertexArtifactError("model configuration has unexpected fields")
    revision = _string(model, "revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise VertexArtifactError("model revision must be an immutable Git commit")
    capabilities = model.get("capabilities")
    if capabilities != ["image", "text", "thinking"]:
        raise VertexArtifactError("model capabilities must pin image, text, and thinking")

    dataset = _mapping(payload, "dataset")
    conversion = _mapping(dataset, "conversion")
    splits = conversion.get("splits")
    if not isinstance(splits, list) or len(splits) != 2:
        raise VertexArtifactError("prepared conversion must contain two split artifacts")
    split_records: dict[str, Mapping[str, object]] = {}
    for record in splits:
        if not isinstance(record, Mapping) or record.get("split") not in {
            "train",
            "validation",
        }:
            raise VertexArtifactError("prepared conversion has an invalid split record")
        split_records[str(record["split"])] = record
    if set(split_records) != {"train", "validation"}:
        raise VertexArtifactError("prepared conversion must contain train and validation")

    prefix_parts = [part.strip("/") for part in (root, "staging", run_id, code_commit) if part]
    prefix = "/".join(prefix_parts)
    base_uri = f"gs://{bucket}/{prefix}"
    artifacts: list[LocalArtifact] = []
    runtime_splits: dict[str, object] = {}
    for split in ("train", "validation"):
        record = split_records[split]
        path_value = record.get("jsonl_path")
        count = record.get("example_count")
        if not isinstance(path_value, str) or not isinstance(count, int) or count < 1:
            raise VertexArtifactError(f"{split} split path or count is invalid")
        sha = record.get("jsonl_sha256")
        uri = f"{base_uri}/inputs/{split}-{sha}.jsonl"
        artifact = _verified_artifact(split, Path(path_value), sha, uri)
        artifacts.append(artifact)
        runtime_splits[split] = {**artifact.to_mapping(), "example_count": count}

    prepared_sha = file_sha256(manifest_path)
    prepared = LocalArtifact(
        "prepared_manifest",
        manifest_path,
        f"{base_uri}/provenance/prepared-manifest-{prepared_sha}.json",
        prepared_sha,
        manifest_path.stat().st_size,
    )
    artifacts.append(prepared)

    output_uri = f"gs://{bucket}/" + "/".join(
        part for part in (root.strip("/"), "runs", run_id) if part
    )
    runtime_payload: dict[str, object] = {
        "runtime_manifest_version": 1,
        "run_id": run_id,
        "code_commit": code_commit,
        "prepared_manifest": prepared.to_mapping(),
        "dataset": {
            "repository": dataset.get("repository"),
            "revision": dataset.get("revision"),
            "splits": runtime_splits,
        },
        "ontology": dict(_mapping(payload, "ontology")),
        "prompt": dict(_mapping(payload, "prompt")),
        "schema": dict(_mapping(payload, "schema")),
        "model": dict(model),
        "training": dict(_mapping(payload, "training")),
        "provider": dict(provider),
        "output_uri": output_uri,
    }
    runtime = _canonical_json(runtime_payload)
    runtime_sha = hashlib.sha256(runtime).hexdigest()
    return VertexStagingPlan(
        run_id=run_id,
        artifacts=tuple(artifacts),
        runtime_manifest=runtime,
        runtime_manifest_uri=f"{base_uri}/runtime-manifest-{runtime_sha}.json",
        runtime_manifest_sha256=runtime_sha,
        output_uri=output_uri,
    )


class GcsArtifactStore:
    """Google Cloud Storage writer with create-only, hash-checked semantics."""

    def __init__(self, client: object | None = None) -> None:
        if client is None:
            from google.cloud import storage

            client = storage.Client()
        self._client = client

    def _blob(self, uri: str) -> object:
        bucket_name, object_name = split_gcs_uri(uri)
        bucket = self._client.bucket(bucket_name)  # type: ignore[attr-defined]
        return bucket.blob(object_name)

    @staticmethod
    def _assert_existing(blob: object, sha256: str, size: int) -> None:
        blob.reload()  # type: ignore[attr-defined]
        metadata = getattr(blob, "metadata", None) or {}
        if metadata.get("edugraph-sha256") != sha256 or getattr(blob, "size", None) != size:
            raise VertexArtifactError("cloud object exists with different content identity")

    def _put(self, uri: str, content: bytes | Path, sha256: str, size: int) -> None:
        blob = self._blob(uri)
        if blob.exists(client=self._client):  # type: ignore[attr-defined]
            self._assert_existing(blob, sha256, size)
            return
        blob.metadata = {"edugraph-sha256": sha256}  # type: ignore[attr-defined]
        try:
            if isinstance(content, Path):
                blob.upload_from_filename(  # type: ignore[attr-defined]
                    str(content), if_generation_match=0, checksum="crc32c"
                )
            else:
                blob.upload_from_string(  # type: ignore[attr-defined]
                    content,
                    content_type="application/json",
                    if_generation_match=0,
                    checksum="crc32c",
                )
        except Exception as error:
            if not blob.exists(client=self._client):  # type: ignore[attr-defined]
                raise VertexArtifactError("failed to create immutable cloud object") from error
            self._assert_existing(blob, sha256, size)

    def put_file(self, artifact: LocalArtifact) -> None:
        self._put(artifact.uri, artifact.path, artifact.sha256, artifact.size)

    def put_bytes(self, uri: str, content: bytes, sha256: str) -> None:
        if hashlib.sha256(content).hexdigest() != sha256:
            raise VertexArtifactError("in-memory artifact does not match its sha256")
        self._put(uri, content, sha256, len(content))


def stage_vertex_run(plan: VertexStagingPlan, writer: ArtifactWriter) -> None:
    """Apply a verified staging plan; safe retries only accept identical objects."""

    for artifact in plan.artifacts:
        writer.put_file(artifact)
    writer.put_bytes(
        plan.runtime_manifest_uri,
        plan.runtime_manifest,
        plan.runtime_manifest_sha256,
    )


def build_vertex_job_config(
    manifest_path: Path,
    staging_record_path: Path,
    image_uri: str,
) -> dict[str, object]:
    """Resolve the fixed first-smoke compute shape after image publication."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    staging = json.loads(staging_record_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(staging, Mapping):
        raise VertexArtifactError("manifest and staging record must be objects")
    run_id = _string(manifest, "run_id")
    code_commit = _string(manifest, "code_commit")
    provider = _mapping(manifest, "provider")
    if staging.get("status") != "staged" or staging.get("run_id") != run_id:
        raise VertexArtifactError("staging record is not complete for this run")
    runtime_uri = staging.get("runtime_manifest_uri")
    runtime_sha = staging.get("runtime_manifest_sha256")
    output_uri = staging.get("output_uri")
    if (
        not isinstance(runtime_uri, str)
        or not isinstance(runtime_sha, str)
        or not _SHA256.fullmatch(runtime_sha)
        or not isinstance(output_uri, str)
    ):
        raise VertexArtifactError("staging record has invalid runtime artifact identity")
    config: dict[str, object] = {
        "provider_id": "gcp_vertex_ai",
        "training_execution_mode": "serverless",
        "job_id": run_id,
        "code_commit": code_commit,
        "project_id": provider.get("project_id"),
        "location": provider.get("location"),
        "display_name": run_id,
        "output_uri": output_uri,
        "service_account": provider.get("service_account"),
        "container": {
            "image_uri": image_uri,
            "command": [],
            "args": ["--runtime-manifest", runtime_uri],
            "environment": {"HF_HOME": "/tmp/huggingface"},
        },
        "compute": {
            "machine_type": "g2-standard-12",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,
            "replica_count": 1,
            "boot_disk_type": "pd-ssd",
            "boot_disk_size_gb": 200,
        },
        "scheduling": {
            "strategy": "FLEX_START",
            "timeout_seconds": 7200,
            "max_wait_seconds": 7200,
        },
        "labels": {
            "environment": "smoke",
            "model": "qwen35-4b",
            "workload": "vlm-training",
        },
    }
    from .providers.vertex import VertexCustomJobSpec

    VertexCustomJobSpec.from_mapping(config)
    return config
