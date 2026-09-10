from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from edugraph_classify.vertex_artifacts import (
    GcsArtifactStore,
    VertexArtifactError,
    build_vertex_job_config,
    build_vertex_staging_plan,
    split_gcs_uri,
    stage_vertex_run,
)


def _manifest(tmp_path: Path) -> Path:
    splits = []
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text('{"messages":[]}\n', encoding="utf-8")
        splits.append(
            {
                "split": split,
                "example_count": 1,
                "jsonl_path": str(path),
                "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "jsonl_size": path.stat().st_size,
                "questions": 1,
                "solutions": 0,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "run_id": "vertex-smoke",
                "code_commit": "a" * 40,
                "dataset": {
                    "repository": "owner/data",
                    "revision": "b" * 40,
                    "conversion": {"splits": splits},
                },
                "ontology": {"version": "1"},
                "prompt": {"prompt_id": "p"},
                "schema": {"schema_id": "s"},
                "model": {
                    "repository": "Qwen/Qwen3.5-4B",
                    "revision": "c" * 40,
                    "created_at": "2026-02-27T14:45:03Z",
                    "capabilities": ["image", "text", "thinking"],
                },
                "provider": {
                    "provider_id": "gcp_vertex_ai",
                    "training_execution_mode": "serverless",
                    "project_id": "project-1",
                    "location": "europe-west4",
                    "bucket_uri": "gs://bucket-1",
                    "service_account": "trainer@project-1.iam.gserviceaccount.com",
                },
                "training": {"method": "supervised_qlora"},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_staging_plan_is_deterministic_and_content_addressed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    first = build_vertex_staging_plan(manifest, "gs://bucket-1")
    second = build_vertex_staging_plan(manifest, "gs://bucket-1")

    assert first == second
    assert first.run_id == "vertex-smoke"
    assert first.output_uri == "gs://bucket-1/runs/vertex-smoke"
    assert first.runtime_manifest_uri.endswith(
        f"runtime-manifest-{first.runtime_manifest_sha256}.json"
    )
    runtime = json.loads(first.runtime_manifest)
    assert runtime["model"]["revision"] == "c" * 40
    assert runtime["dataset"]["splits"]["train"]["uri"].startswith(
        "gs://bucket-1/staging/vertex-smoke/" + "a" * 40
    )
    assert first.to_mapping()["runtime_manifest"]["size"] > 0


def test_staging_plan_rejects_mismatched_inputs_and_identity(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    with pytest.raises(VertexArtifactError, match="bucket"):
        build_vertex_staging_plan(manifest, "gs://different-bucket")

    payload["dataset"]["conversion"]["splits"][0]["jsonl_sha256"] = "d" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VertexArtifactError, match="sha256"):
        build_vertex_staging_plan(manifest, "gs://bucket-1")

    payload["dataset"]["conversion"]["splits"][0]["jsonl_sha256"] = hashlib.sha256(
        (tmp_path / "train.jsonl").read_bytes()
    ).hexdigest()
    payload["model"]["capabilities"] = ["text"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VertexArtifactError, match="capabilities"):
        build_vertex_staging_plan(manifest, "gs://bucket-1")


def test_staging_plan_rejects_all_structural_identity_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    baseline = json.loads(manifest.read_text(encoding="utf-8"))

    def check(mutate: object, message: str) -> None:
        payload = deepcopy(baseline)
        mutate(payload)  # type: ignore[operator]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(VertexArtifactError, match=message):
            build_vertex_staging_plan(manifest, "gs://bucket-1")

    cases = [
        (lambda value: value.update(manifest_version=2), "version"),
        (lambda value: value.update(run_id=" Bad "), "run_id"),
        (lambda value: value.update(code_commit="short"), "code_commit"),
        (lambda value: value.update(provider=[]), "provider must be an object"),
        (lambda value: value["provider"].update(extra=True), "unexpected fields"),
        (lambda value: value["provider"].update(provider_id="other"), "not a Vertex"),
        (lambda value: value["model"].update(extra=True), "model configuration"),
        (lambda value: value["model"].update(revision="main"), "immutable"),
        (lambda value: value["dataset"]["conversion"].update(splits=[]), "two split"),
        (
            lambda value: value["dataset"]["conversion"]["splits"][0].update(split="test"),
            "invalid split",
        ),
        (
            lambda value: value["dataset"]["conversion"]["splits"][1].update(split="train"),
            "train and validation",
        ),
        (
            lambda value: value["dataset"]["conversion"]["splits"][0].update(
                example_count=0
            ),
            "path or count",
        ),
        (
            lambda value: value["dataset"]["conversion"]["splits"][0].update(
                jsonl_sha256="bad"
            ),
            "invalid recorded",
        ),
        (
            lambda value: value["dataset"]["conversion"]["splits"][0].update(
                jsonl_path=str(tmp_path / "missing.jsonl")
            ),
            "does not exist",
        ),
    ]
    for mutate, message in cases:
        check(mutate, message)

    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(VertexArtifactError, match="version"):
        build_vertex_staging_plan(manifest, "gs://bucket-1")

def test_gcs_uri_and_staging_application_are_strict(tmp_path: Path) -> None:
    assert split_gcs_uri("gs://bucket-1/path/item") == ("bucket-1", "path/item")
    assert split_gcs_uri("gs://bucket-1", require_object=False) == ("bucket-1", "")
    with pytest.raises(VertexArtifactError, match="valid"):
        split_gcs_uri("https://bucket-1/item")
    with pytest.raises(VertexArtifactError, match="object"):
        split_gcs_uri("gs://bucket-1")

    plan = build_vertex_staging_plan(_manifest(tmp_path), "gs://bucket-1")

    class Writer:
        def __init__(self) -> None:
            self.files: list[str] = []
            self.bytes: list[str] = []

        def put_file(self, artifact: object) -> None:
            self.files.append(artifact.name)  # type: ignore[attr-defined]

        def put_bytes(self, uri: str, content: bytes, sha256: str) -> None:
            assert hashlib.sha256(content).hexdigest() == sha256
            self.bytes.append(uri)

    writer = Writer()
    stage_vertex_run(plan, writer)
    assert writer.files == ["train", "validation", "prepared_manifest"]
    assert writer.bytes == [plan.runtime_manifest_uri]


class FakeBlob:
    def __init__(self, *, exists: bool = False, sha256: str | None = None, size: int = 0) -> None:
        self.present = exists
        self.metadata = {} if sha256 is None else {"edugraph-sha256": sha256}
        self.size = size
        self.uploads = 0
        self.fail_upload = False

    def exists(self, **kwargs: object) -> bool:
        return self.present

    def reload(self) -> None:
        pass

    def upload_from_filename(self, path: str, **kwargs: object) -> None:
        self.uploads += 1
        self.size = Path(path).stat().st_size
        self.present = True
        if self.fail_upload:
            raise RuntimeError("race")

    def upload_from_string(self, content: bytes, **kwargs: object) -> None:
        self.uploads += 1
        self.size = len(content)
        self.present = True
        if self.fail_upload:
            raise RuntimeError("race")


class FakeBucket:
    def __init__(self) -> None:
        self.blobs: dict[str, FakeBlob] = {}

    def blob(self, name: str) -> FakeBlob:
        return self.blobs.setdefault(name, FakeBlob())


class FakeStorageClient:
    def __init__(self) -> None:
        self.buckets: dict[str, FakeBucket] = {}

    def bucket(self, name: str) -> FakeBucket:
        return self.buckets.setdefault(name, FakeBucket())


def test_gcs_store_creates_reuses_and_rejects_conflicting_objects(tmp_path: Path) -> None:
    plan = build_vertex_staging_plan(_manifest(tmp_path), "gs://bucket-1")
    client = FakeStorageClient()
    store = GcsArtifactStore(client)
    artifact = plan.artifacts[0]

    store.put_file(artifact)
    blob = client.bucket("bucket-1").blob(split_gcs_uri(artifact.uri)[1])
    assert blob.uploads == 1
    store.put_file(artifact)
    assert blob.uploads == 1

    blob.metadata = {"edugraph-sha256": "f" * 64}
    with pytest.raises(VertexArtifactError, match="different"):
        store.put_file(artifact)

    content = b"{}\n"
    digest = hashlib.sha256(content).hexdigest()
    store.put_bytes("gs://bucket-1/runtime.json", content, digest)
    with pytest.raises(VertexArtifactError, match="does not match"):
        store.put_bytes("gs://bucket-1/other.json", content, "0" * 64)


def test_gcs_store_accepts_identical_create_race_and_reports_failed_create(tmp_path: Path) -> None:
    plan = build_vertex_staging_plan(_manifest(tmp_path), "gs://bucket-1")
    client = FakeStorageClient()
    store = GcsArtifactStore(client)
    artifact = plan.artifacts[0]
    blob = client.bucket("bucket-1").blob(split_gcs_uri(artifact.uri)[1])
    blob.fail_upload = True
    blob.metadata = {"edugraph-sha256": artifact.sha256}
    store.put_file(artifact)

    missing = client.bucket("bucket-1").blob("missing")
    missing.fail_upload = True
    original_exists = missing.exists

    def never_exists(**kwargs: object) -> bool:
        original_exists(**kwargs)
        return False

    missing.exists = never_exists  # type: ignore[method-assign]
    with pytest.raises(VertexArtifactError, match="failed to create"):
        store.put_bytes("gs://bucket-1/missing", b"x", hashlib.sha256(b"x").hexdigest())


def test_job_configuration_is_derived_from_completed_staging(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    plan = build_vertex_staging_plan(manifest, "gs://bucket-1")
    staging = tmp_path / "staging.json"
    staging.write_text(
        json.dumps({"status": "staged", **plan.to_mapping(), "runtime_manifest_uri": plan.runtime_manifest_uri, "runtime_manifest_sha256": plan.runtime_manifest_sha256}),
        encoding="utf-8",
    )
    image = "europe-west4-docker.pkg.dev/project/repo/vlm@sha256:" + "e" * 64
    config = build_vertex_job_config(manifest, staging, image)

    assert config["job_id"] == "vertex-smoke"
    assert config["container"]["args"] == [  # type: ignore[index]
        "--runtime-manifest",
        plan.runtime_manifest_uri,
    ]
    assert config["compute"]["accelerator_type"] == "NVIDIA_L4"  # type: ignore[index]
    assert config["scheduling"]["strategy"] == "FLEX_START"  # type: ignore[index]

    diagnostic = build_vertex_job_config(
        manifest, staging, image, diagnostic_one_batch=True
    )
    assert diagnostic["container"]["args"] == [  # type: ignore[index]
        "--runtime-manifest",
        plan.runtime_manifest_uri,
        "--diagnostic-one-batch",
    ]

    staging.write_text('{"status":"pending"}', encoding="utf-8")
    with pytest.raises(VertexArtifactError, match="not complete"):
        build_vertex_job_config(manifest, staging, image)
