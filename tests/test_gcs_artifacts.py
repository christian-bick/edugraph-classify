"""Offline checks for the GCS primitives used by RunPod staging and recovery."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from edugraph_classify.gcs_artifacts import (
    GcsArtifactError,
    GcsArtifactStore,
    LocalArtifact,
    split_gcs_uri,
)


class FakeBlob:
    def __init__(self) -> None:
        self.present = False
        self.metadata: dict[str, str] = {}
        self.size = 0
        self.uploads = 0
        self.fail_after_create = False
        self.fail_without_create = False

    def exists(self, **kwargs: object) -> bool:
        return self.present

    def reload(self) -> None:
        pass

    def _upload(self, content: bytes) -> None:
        self.uploads += 1
        if not self.fail_without_create:
            self.present = True
            self.size = len(content)
        if self.fail_after_create or self.fail_without_create:
            raise RuntimeError("upload interrupted")

    def upload_from_filename(self, path: str, **kwargs: object) -> None:
        self._upload(Path(path).read_bytes())

    def upload_from_string(self, content: bytes, **kwargs: object) -> None:
        self._upload(content)


class FakeClient:
    def __init__(self) -> None:
        self.blobs: dict[tuple[str, str], FakeBlob] = {}

    def bucket(self, name: str) -> FakeClient:
        self.bucket_name = name
        return self

    def blob(self, name: str) -> FakeBlob:
        return self.blobs.setdefault((self.bucket_name, name), FakeBlob())


def test_gcs_uri_requires_a_valid_object_name() -> None:
    assert split_gcs_uri("gs://bucket-1/path/item") == ("bucket-1", "path/item")
    assert split_gcs_uri("gs://bucket-1", require_object=False) == ("bucket-1", "")
    with pytest.raises(GcsArtifactError, match="valid gs://"):
        split_gcs_uri("https://bucket-1/path/item")
    with pytest.raises(GcsArtifactError, match="name an object"):
        split_gcs_uri("gs://bucket-1")


def test_immutable_file_upload_reuses_identity_and_rejects_conflict(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.tar"
    path.write_bytes(b"checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = LocalArtifact("checkpoint", path, "gs://bucket-1/checkpoint.tar", digest, path.stat().st_size)
    assert artifact.to_mapping() == {
        "name": "checkpoint", "uri": artifact.uri, "sha256": digest, "size": 10,
    }

    client = FakeClient()
    store = GcsArtifactStore(client)
    store.put_file(artifact)
    blob = client.blobs[("bucket-1", "checkpoint.tar")]
    assert blob.uploads == 1
    assert blob.metadata == {"edugraph-sha256": digest}
    store.put_file(artifact)
    assert blob.uploads == 1

    blob.size += 1
    with pytest.raises(GcsArtifactError, match="different content identity"):
        store.put_file(artifact)


def test_bytes_upload_checks_digest_and_handles_create_races() -> None:
    client = FakeClient()
    store = GcsArtifactStore(client)
    content = b'{"completed":true}\n'
    digest = hashlib.sha256(content).hexdigest()
    with pytest.raises(GcsArtifactError, match="does not match"):
        store.put_bytes("gs://bucket-1/bad.json", content, "0" * 64)

    race = client.bucket("bucket-1").blob("race.json")
    race.fail_after_create = True
    store.put_bytes("gs://bucket-1/race.json", content, digest)
    assert race.present and race.uploads == 1
    store.put_bytes("gs://bucket-1/race.json", content, digest)
    assert race.uploads == 1

    failed = client.bucket("bucket-1").blob("failed.json")
    failed.fail_without_create = True
    with pytest.raises(GcsArtifactError, match="failed to create"):
        store.put_bytes("gs://bucket-1/failed.json", content, digest)
