"""Immutable Google Cloud Storage artifacts shared by training executors."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


class GcsArtifactError(ValueError):
    """Raised when an artifact URI is invalid or cloud content conflicts."""


_GCS_ROOT = re.compile(r"^gs://([a-z0-9][a-z0-9._-]{1,220})(?:/([^\s]+))?$")


def split_gcs_uri(uri: str, *, require_object: bool = True) -> tuple[str, str]:
    match = _GCS_ROOT.fullmatch(uri)
    if match is None:
        raise GcsArtifactError("artifact URI must be a valid gs:// URI")
    bucket, object_name = match.group(1), match.group(2) or ""
    if require_object and not object_name:
        raise GcsArtifactError("artifact URI must name an object")
    return bucket, object_name


@dataclass(frozen=True, slots=True)
class LocalArtifact:
    """One local file and its content-addressed destination."""

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


class GcsArtifactStore:
    """Create-only GCS writer that accepts retries with identical content."""

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
            raise GcsArtifactError("cloud object exists with different content identity")

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
                raise GcsArtifactError("failed to create immutable cloud object") from error
            self._assert_existing(blob, sha256, size)

    def put_file(self, artifact: LocalArtifact) -> None:
        self._put(artifact.uri, artifact.path, artifact.sha256, artifact.size)

    def put_bytes(self, uri: str, content: bytes, sha256: str) -> None:
        if hashlib.sha256(content).hexdigest() != sha256:
            raise GcsArtifactError("in-memory artifact does not match its sha256")
        self._put(uri, content, sha256, len(content))
