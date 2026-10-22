"""Stage the verified GGUF candidate as a minimal Cloud Build Docker context."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Candidate:
    gcs_uri: str
    archive_bytes: int
    archive_sha256: str
    member_sha256: Mapping[str, str]


CANDIDATE = Candidate(
    gcs_uri=(
        "gs://edugraph-classify/runpod/inference-models/"
        "edugraph-20261004-qwen38-27b-gguf-benchmark-v4/"
        "7fc4a0f8fc21761615751107e92b44ffe4f9f75a9cde3aaac6613e4d51c33c72.tar"
        "#1791147881432887"
    ),
    archive_bytes=17_478_901_760,
    archive_sha256="7fc4a0f8fc21761615751107e92b44ffe4f9f75a9cde3aaac6613e4d51c33c72",
    member_sha256={
        "model-Q4_K_M.gguf": "bd2ec1357e84b27b58ec0b1c83e143028f9b54e42989dc9f3db7c59aed870639",
        "mmproj-BF16.gguf": "3248a0391bed86eef3842121dab70d4657da244bde47ae02ce37580351bd0138",
        "chat_template.jinja": "4813f6c608147061ba44f17b01aa344de7b5036cb97bb39200a3b2bb6e26a0c5",
        "prompt.json": "17e336fff2c539d9413defc12a8578e770e220f272e1ef92f90dfedc38c7c949",
        "closed_schema.json": "a59c3cbed3a817bc20fec04c2d13c003b9be6360b0f8d3348036da884376f5b6",
        "manifest.json": "92e6a5214a18663ca1d81bea5e142db995d75a6f64daf107f6ebca819a62ff4e",
    },
)

IMAGE_FILES = ("model-Q4_K_M.gguf", "mmproj-BF16.gguf", "chat_template.jinja")
MODEL_HF_REVISION = "477e3caf32eced2a9e002ead1edccc9a5c458f0c"
CHUNK_BYTES = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source, output: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with output.open("xb") as target:
        while chunk := source.read(CHUNK_BYTES):
            target.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"candidate member SHA-256 mismatch: {output.name}")


def prepare_context(
    archive_path: Path,
    output_dir: Path,
    *,
    candidate: Candidate = CANDIDATE,
    dockerfile: Path | None = None,
) -> None:
    """Verify the full archive and every source member before staging three files."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if archive_path.stat().st_size != candidate.archive_bytes:
        raise ValueError("candidate archive byte count mismatch")
    if sha256_file(archive_path) != candidate.archive_sha256:
        raise ValueError("candidate archive SHA-256 mismatch")

    stage = Path(tempfile.mkdtemp(prefix="gguf-build-", dir=output_dir.parent))
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            by_name = {member.name: member for member in members}
            expected_names = frozenset((*candidate.member_sha256, "candidate.json"))
            if (len(by_name) != len(members) or frozenset(by_name) != expected_names
                    or any(member.type != tarfile.REGTYPE for member in members)):
                raise ValueError("candidate archive members are not the pinned regular files")
            for name, expected_sha256 in candidate.member_sha256.items():
                source = archive.extractfile(by_name[name])
                if source is None:
                    raise ValueError(f"candidate member cannot be read: {name}")
                with source:
                    if name in IMAGE_FILES:
                        _copy_verified(source, stage / name, expected_sha256)
                    else:
                        digest = hashlib.sha256()
                        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                            digest.update(chunk)
                        if digest.hexdigest() != expected_sha256:
                            raise ValueError(f"candidate member SHA-256 mismatch: {name}")
        shutil.copyfile(dockerfile or Path(__file__).with_name("Dockerfile"), stage / "Dockerfile")
        os.replace(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-hf-revision", required=True)
    args = parser.parse_args(argv)
    if args.model_hf_revision != MODEL_HF_REVISION:
        parser.error("--model-hf-revision must match the published Hub release")

    workspace = Path(__file__).resolve().parent
    archive_path = workspace / "candidate.tar"
    output_dir = workspace / "build-context"
    if archive_path.exists() or output_dir.exists():
        raise FileExistsError("Cloud Build workspace must not contain a prior candidate or context")
    try:
        subprocess.run(["gcloud", "storage", "cp", CANDIDATE.gcs_uri, str(archive_path)], check=True)
        prepare_context(archive_path, output_dir, candidate=CANDIDATE)
    finally:
        archive_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
