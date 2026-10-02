"""Immutable, verified bundles shared by self-hosted training and recovery."""
from __future__ import annotations

import json
import os
import shutil
import tarfile
from pathlib import Path

from .dataset import file_sha256
from .learning_data import safe_relative, write_json
from .vertex_artifacts import GcsArtifactStore, LocalArtifact, split_gcs_uri


def pack_files(root: Path, names: list[str], output: Path) -> str:
    """Deterministic uncompressed tar; large tensor checkpoints already compress poorly."""
    with tarfile.open(output, "w") as archive:
        for name in sorted(set(names)):
            source = root / safe_relative(name)
            if source.is_symlink() or not source.resolve().is_relative_to(root.resolve()):
                raise ValueError("bundle cannot contain links or escaped paths")
            info = tarfile.TarInfo(name)
            info.size, info.mode = source.stat().st_size, 0o600
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    return file_sha256(output)


def unpack_verified(archive_path: Path, digest: str, destination: Path) -> None:
    if file_sha256(archive_path) != digest:
        raise ValueError("bundle checksum mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        names = [safe_relative(member.name) for member in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError("bundle must contain unique regular files only")
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


class GcsBundles:
    def __init__(self, client):
        self.client = client
        self.writer = GcsArtifactStore(client)

    @classmethod
    def from_environment(cls):
        from google.cloud import storage
        from google.oauth2 import service_account

        secret = os.environ.get("EDUGRAPH_GCS_CREDENTIALS_JSON")
        if not secret:
            return cls(storage.Client())  # Local ADC or a mounted workload identity.
        info = json.loads(secret)
        if info.get("type") != "service_account":
            raise ValueError("Pod credential secret must be a service account, not personal ADC")
        credentials = service_account.Credentials.from_service_account_info(info)
        return cls(storage.Client(project=info["project_id"], credentials=credentials))

    def publish(self, path: Path, root_uri: str, run_id: str) -> dict:
        digest = file_sha256(path)
        uri = root_uri.rstrip("/") + "/" + digest + ".tar"
        self.writer.put_file(LocalArtifact(path.name, path, uri, digest, path.stat().st_size))
        return {"run_id": run_id, "uri": uri, "sha256": digest, "bytes": path.stat().st_size}

    def download(self, reference: dict, output: Path) -> None:
        bucket, key = split_gcs_uri(reference["uri"])
        self.client.bucket(bucket).blob(key).download_to_filename(str(output), checksum="crc32c")
        if file_sha256(output) != reference["sha256"]:
            raise ValueError("downloaded bundle checksum mismatch")

    def complete(self, reference: dict, marker_uri: str) -> None:
        """A completion marker is written only after the full immutable bundle exists."""
        import hashlib
        content = (json.dumps(reference, sort_keys=True) + "\n").encode()
        self.writer.put_bytes(marker_uri, content, hashlib.sha256(content).hexdigest())

    def read_json(self, uri: str) -> dict | None:
        bucket, key = split_gcs_uri(uri)
        blob = self.client.bucket(bucket).blob(key)
        return json.loads(blob.download_as_bytes()) if blob.exists() else None


def publish_snapshot(directory: Path, scratch: Path, store, root_uri: str, run_id: str) -> dict:
    scratch.parent.mkdir(parents=True, exist_ok=True)
    names = [p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()]
    pack_files(directory, names, scratch)
    reference = store.publish(scratch, root_uri, run_id)
    write_json(directory / "external.json", reference)
    scratch.unlink()
    return reference
