"""Offline checks for the generation-pinned Cloud Build model context."""

import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "containers" / "gcp-gguf-inference" / "prepare_context.py"
SPEC = importlib.util.spec_from_file_location("gcp_gguf_build_context", HELPER)
assert SPEC is not None and SPEC.loader is not None
build_context = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_context
SPEC.loader.exec_module(build_context)

SOURCE_BYTES = {
    "model-Q4_K_M.gguf": b"language model",
    "mmproj-BF16.gguf": b"vision projector",
    "chat_template.jinja": b"{{ messages }}",
    "prompt.json": b"{}",
    "closed_schema.json": b"{}",
    "manifest.json": b"{}",
    "candidate.json": b"{}",
}


def _archive(tmp_path: Path, names: list[str] | None = None, *, link: bool = False):
    path = tmp_path / "candidate.tar"
    with tarfile.open(path, "w") as archive:
        for name in names or sorted(SOURCE_BYTES):
            data = SOURCE_BYTES.get(name, b"extra")
            info = tarfile.TarInfo(name)
            info.mode = 0o600
            if link and name == "model-Q4_K_M.gguf":
                info.type = tarfile.SYMTYPE
                info.linkname = "mmproj-BF16.gguf"
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    candidate = build_context.Candidate(
        gcs_uri="gs://fake/candidate.tar#123",
        archive_bytes=path.stat().st_size,
        archive_sha256=build_context.sha256_file(path),
        member_sha256={
            name: hashlib.sha256(data).hexdigest()
            for name, data in SOURCE_BYTES.items() if name != "candidate.json"
        },
    )
    return path, candidate


def test_prepare_context_stages_only_verified_image_inputs(tmp_path):
    archive, candidate = _archive(tmp_path)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_bytes(b"FROM scratch\n")
    output = tmp_path / "context"

    build_context.prepare_context(archive, output, candidate=candidate, dockerfile=dockerfile)

    assert {p.name for p in output.iterdir()} == {*build_context.IMAGE_FILES, "Dockerfile"}
    for name in build_context.IMAGE_FILES:
        assert (output / name).read_bytes() == SOURCE_BYTES[name]
    assert (output / "Dockerfile").read_bytes() == b"FROM scratch\n"


@pytest.mark.parametrize("change, message", [
    ("size", "byte count"),
    ("archive_sha256", "archive SHA-256"),
    ("member_sha256", "member SHA-256"),
])
def test_prepare_context_rejects_changed_hashes_and_size(tmp_path, change, message):
    archive, candidate = _archive(tmp_path)
    fields = dict(vars(candidate))
    if change == "size":
        fields["archive_bytes"] += 1
    elif change == "archive_sha256":
        fields["archive_sha256"] = "0" * 64
    else:
        fields["member_sha256"] = dict(fields["member_sha256"])
        fields["member_sha256"]["mmproj-BF16.gguf"] = "0" * 64

    with pytest.raises(ValueError, match=message):
        build_context.prepare_context(
            archive, tmp_path / "context", candidate=build_context.Candidate(**fields)
        )
    assert not (tmp_path / "context").exists()


@pytest.mark.parametrize("names, link", [
    (sorted(SOURCE_BYTES) + ["extra.txt"], False),
    (sorted(SOURCE_BYTES) + ["model-Q4_K_M.gguf"], False),
    (sorted(SOURCE_BYTES), True),
    ([name for name in sorted(SOURCE_BYTES) if name != "candidate.json"], False),
])
def test_prepare_context_rejects_unexpected_tar_members(tmp_path, names, link):
    archive, candidate = _archive(tmp_path, names, link=link)
    with pytest.raises(ValueError, match="pinned regular files"):
        build_context.prepare_context(archive, tmp_path / "context", candidate=candidate)
    assert not (tmp_path / "context").exists()


def test_build_python_step_uses_pinned_python_capable_image():
    config = (HELPER.parent / "cloudbuild.yaml").read_text()
    assert re.search(
        r"name: gcr\.io/google\.com/cloudsdktool/google-cloud-cli:slim@sha256:[0-9a-f]{64}",
        config,
    )
    assert "entrypoint: python3" in config


def test_build_pins_match_public_release_recipe():
    recipe = json.loads((ROOT / "experiments" / "hf-gguf-release-qwen38-27b-v1.json").read_text())
    publication = json.loads((ROOT / "experiments" / "hf-gguf-publication-qwen38-27b-v1.json").read_text())
    candidate = build_context.CANDIDATE
    assert build_context.MODEL_HF_REVISION == publication["revision"]
    assert (
        f'&& test "$MODEL_HF_REVISION" = "{publication["revision"]}"'
        in (HELPER.parent / "Dockerfile").read_text()
    )
    assert candidate.archive_bytes == recipe["candidate"]["archive_bytes"]
    assert candidate.archive_sha256 == recipe["candidate"]["archive_sha256"]
    assert candidate.member_sha256 == recipe["files_sha256"]
    assert candidate.gcs_uri.endswith(
        f"/{candidate.archive_sha256}.tar#1791147881432887"
    )


@pytest.mark.parametrize("revision", ["main", "a" * 40])
def test_cli_requires_published_hub_commit_before_downloading(tmp_path, monkeypatch, revision):
    monkeypatch.setattr(build_context, "__file__", str(tmp_path / "prepare_context.py"))
    monkeypatch.setattr(
        build_context.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("download must not start"),
    )
    with pytest.raises(SystemExit):
        build_context.main(["--model-hf-revision", revision])
    assert not (tmp_path / "candidate.tar").exists()


def test_cli_uses_generation_pin_and_cleans_download(tmp_path, monkeypatch):
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    archive, candidate = _archive(fixture_dir)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Dockerfile").write_bytes(b"FROM scratch\n")
    monkeypatch.setattr(build_context, "__file__", str(workspace / "prepare_context.py"))
    monkeypatch.setattr(build_context, "CANDIDATE", candidate)
    calls = []

    def fake_download(args, *, check):
        calls.append((args, check))
        shutil.copyfile(archive, args[-1])

    monkeypatch.setattr(build_context.subprocess, "run", fake_download)
    assert build_context.main(["--model-hf-revision", build_context.MODEL_HF_REVISION]) == 0
    assert calls == [(
        ["gcloud", "storage", "cp", "gs://fake/candidate.tar#123",
         str(workspace / "candidate.tar")], True
    )]
    assert not (workspace / "candidate.tar").exists()
    assert (workspace / "build-context" / "Dockerfile").read_bytes() == b"FROM scratch\n"


def test_cli_cleans_partial_download_on_gcloud_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(build_context, "__file__", str(tmp_path / "prepare_context.py"))

    def fake_download(args, *, check):
        Path(args[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(build_context.subprocess, "run", fake_download)
    with pytest.raises(subprocess.CalledProcessError):
        build_context.main(["--model-hf-revision", build_context.MODEL_HF_REVISION])
    assert not (tmp_path / "candidate.tar").exists()
    assert not (tmp_path / "build-context").exists()
