"""Offline integrity and injection checks for the generation-pinned build context."""

from dataclasses import replace
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
PROFILE_PATH = HELPER.parent / "profiles" / "qwen38-27b-k6-v1.json"
PROFILE = build_context.load_profile(PROFILE_PATH)

SOURCE_BYTES = {
    "model-Q4_K_M.gguf": b"language model",
    "mmproj-BF16.gguf": b"vision projector",
    "chat_template.jinja": b"{{ messages }}",
    "prompt.json": b"{}",
    "closed_schema.json": b"{}",
    "manifest.json": b"{}",
    "candidate.json": b"{}",
}


def _archive(tmp_path: Path, names=None, *, link=False):
    path = tmp_path / "candidate.tar"
    with tarfile.open(path, "w") as archive:
        for name in names or sorted(SOURCE_BYTES):
            data = SOURCE_BYTES.get(name, b"extra")
            info = tarfile.TarInfo(name)
            if link and name == "model-Q4_K_M.gguf":
                info.type, info.linkname = tarfile.SYMTYPE, "mmproj-BF16.gguf"
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    digest = build_context.sha256_file(path)
    candidate = build_context.Candidate(
        gcs_uri=f"gs://edugraph-classify/runpod/inference-models/fake/{digest}.tar#123",
        archive_bytes=path.stat().st_size, archive_sha256=digest,
        member_sha256={name: hashlib.sha256(data).hexdigest()
                       for name, data in SOURCE_BYTES.items() if name != "candidate.json"},
    )
    return path, replace(PROFILE, candidate=candidate)


def _profile_file(tmp_path, data):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_prepare_context_stages_only_verified_inputs_and_renders_settings(tmp_path):
    archive, profile = _archive(tmp_path)
    output = tmp_path / "context"
    build_context.prepare_context(archive, output, profile=profile)
    assert {p.name for p in output.iterdir()} == {*build_context.IMAGE_FILES, "Dockerfile"}
    for name in build_context.IMAGE_FILES:
        assert (output / name).read_bytes() == SOURCE_BYTES[name]
    rendered = (output / "Dockerfile").read_text()
    assert "@@" not in rendered
    command = json.loads(next(line[4:] for line in rendered.splitlines() if line.startswith("CMD ")))
    for flag in ("--ctx-size", "--batch-size", "--ubatch-size"):
        assert command[command.index(flag) + 1] == "4096"
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--fit") + 1] == "off"
    assert command[command.index("--n-gpu-layers") + 1] == "all"
    assert profile.hub_revision in rendered


@pytest.mark.parametrize("change, message", [
    ("size", "byte count"), ("archive_sha256", "archive SHA-256"),
    ("model-Q4_K_M.gguf", "member SHA-256"), ("prompt.json", "member SHA-256"),
])
def test_prepare_rejects_corruption_and_leaves_no_context(tmp_path, change, message):
    archive, profile = _archive(tmp_path)
    candidate = profile.candidate
    if change == "size":
        candidate = replace(candidate, archive_bytes=candidate.archive_bytes + 1)
    elif change == "archive_sha256":
        candidate = replace(candidate, archive_sha256="0" * 64)
    else:
        candidate = replace(candidate, member_sha256={**candidate.member_sha256, change: "0" * 64})
    with pytest.raises(ValueError, match=message):
        build_context.prepare_context(archive, tmp_path / "context", profile=replace(profile, candidate=candidate))
    assert not (tmp_path / "context").exists()
    assert not list(tmp_path.glob("gguf-build-*"))


@pytest.mark.parametrize("names, link", [
    (sorted(SOURCE_BYTES) + ["extra.txt"], False),
    (sorted(SOURCE_BYTES) + ["../escape"], False),
    (sorted(SOURCE_BYTES) + ["model-Q4_K_M.gguf"], False),
    (sorted(SOURCE_BYTES), True),
    ([name for name in sorted(SOURCE_BYTES) if name != "candidate.json"], False),
])
def test_prepare_rejects_unsafe_or_unexpected_tar_members(tmp_path, names, link):
    archive, profile = _archive(tmp_path, names, link=link)
    with pytest.raises(ValueError, match="pinned regular files"):
        build_context.prepare_context(archive, tmp_path / "context", profile=profile)
    assert not (tmp_path / "context").exists()


@pytest.mark.parametrize("section, key, value", [
    ("publication", "revision", "main"),
    ("publication", "revision", "a" * 40 + "\nRUN false"),
    ("publication", "repository", "someone/other"),
    ("runtime", "image", "image:latest"),
    ("runtime", "image", PROFILE.runtime_image + "\nRUN false"),
    ("runtime", "llama_cpp_commit", "HEAD"),
    ("candidate", "archive_sha256", "x" * 64),
    ("candidate", "archive_bytes", True),
    ("candidate", "archive_bytes", -1),
    ("candidate", "gcs_uri", PROFILE.candidate.gcs_uri.split("#")[0]),
    ("candidate", "gcs_uri", PROFILE.candidate.gcs_uri.replace("edugraph-classify/", "other-bucket/")),
    ("candidate", "gcs_uri", PROFILE.candidate.gcs_uri.replace("/inference-models/", "/inference-models/../")),
    ("candidate", "gcs_uri", PROFILE.candidate.gcs_uri + "; echo unsafe"),
    ("inference", "context_tokens", "4096; false"),
    ("inference", "context_tokens", 8192),
    ("inference", "batch_size", 2048),
    ("inference", "ubatch_size", True),
    ("inference", "ubatch_size", 0),
    ("evidence", "recipe_path", "../outside.json"),
    ("evidence", "publication_sha256", "main"),
])
def test_profile_rejects_floating_unsafe_or_inconsistent_values(tmp_path, section, key, value):
    data = json.loads(PROFILE_PATH.read_text())
    data[section][key] = value
    with pytest.raises(ValueError):
        build_context.load_profile(_profile_file(tmp_path, data))


def test_profile_rejects_unexpected_fields_members_and_duplicate_json(tmp_path):
    for section, key in ((None, "command"), ("candidate", "unexpected"), ("inference", "command")):
        data = json.loads(PROFILE_PATH.read_text())
        (data if section is None else data[section])[key] = "unsafe"
        with pytest.raises(ValueError, match="fields"):
            build_context.load_profile(_profile_file(tmp_path, data))
    data = json.loads(PROFILE_PATH.read_text())
    data["candidate"]["member_sha256"]["../escape"] = "0" * 64
    with pytest.raises(ValueError, match="member hashes"):
        build_context.load_profile(_profile_file(tmp_path, data))
    path = tmp_path / "duplicate.json"
    path.write_text('{"kind":"first","kind":"second"}')
    with pytest.raises(ValueError, match="duplicate"):
        build_context.load_profile(path)


@pytest.mark.parametrize("path", sorted((HELPER.parent / "profiles").glob("*.json")), ids=lambda path: path.name)
def test_profiles_bind_to_published_release_and_proven_runtime(path):
    profile = build_context.load_profile(path)
    build_context.verify_release_evidence(profile, ROOT)
    canary = json.loads((ROOT / "experiments/imagine-gguf-gcp-canary-20261006.json").read_text())
    assert profile.runtime_image == build_context.RUNTIME_REPOSITORY + "@" + canary["image"]["digest"]
    assert profile.llama_cpp_commit == "0faee5004297c3bcfa40b7bf11750127b8c1fd7d"
    if path.name == "qwen38-27b-k6-v1.json":
        assert (profile.context_tokens, profile.batch_size, profile.ubatch_size) == (4096, 4096, 4096)
        assert profile.candidate.gcs_uri.endswith("#1792947143663173")
    elif path.name == "qwen38-27b-v1.json":
        assert (profile.context_tokens, profile.batch_size, profile.ubatch_size) == (2048, 2048, 1024)


def test_release_evidence_rejects_pin_and_evidence_changes():
    for profile in (replace(PROFILE, hub_revision="a" * 40),
                    replace(PROFILE, llama_cpp_commit="b" * 40),
                    replace(PROFILE, context_tokens=2048),
                    replace(PROFILE, candidate=replace(PROFILE.candidate, gcs_uri=PROFILE.candidate.gcs_uri.replace("-v2/", "-other/")))):
        with pytest.raises(ValueError, match="does not match"):
            build_context.verify_release_evidence(profile, ROOT)
    changed = replace(PROFILE, evidence={**PROFILE.evidence, "recipe_sha256": "0" * 64})
    with pytest.raises(ValueError, match="evidence SHA-256"):
        build_context.verify_release_evidence(changed, ROOT)


def test_runtime_copy_excludes_old_weight_layers_and_checks_binary():
    rendered = build_context.render_dockerfile(HELPER.parent / "Dockerfile", PROFILE)
    stages = [line for line in rendered.splitlines() if line.startswith("FROM ")]
    assert stages[0] == f"FROM {PROFILE.runtime_image} AS verified-runtime"
    assert stages[-1].startswith("FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:")
    assert "COPY --from=verified-runtime /opt/llama.cpp/bin /opt/llama.cpp/bin" in rendered
    assert "llama-server --version" in rendered and PROFILE.llama_cpp_commit[:7] in rendered
    assert "llama-server --help" in rendered
    assert "--ctx-size --batch-size --ubatch-size --mmproj --fit" in rendered
    assert "bd2ec1357e84b27b58ec0b1c83e143028f9b54e42989dc9f3db7c59aed870639" not in rendered


def test_template_rejects_unknown_or_missing_fields(tmp_path):
    template = tmp_path / "Dockerfile"
    template.write_text("FROM @@UNKNOWN@@\n")
    with pytest.raises(ValueError, match="template fields"):
        build_context.render_dockerfile(template, PROFILE)


def test_build_configuration_pins_builders_and_keeps_resource_limit():
    config = (HELPER.parent / "cloudbuild.yaml").read_text()
    assert re.search(r"name: gcr\.io/google\.com/cloudsdktool/google-cloud-cli:slim@sha256:[0-9a-f]{64}", config)
    assert "gcr.io/cloud-builders/docker@sha256:001fb4a870a84485cf198c80002f7af42f6456f2dbc261d70a0b5f0111470df4" in config
    assert "'${_RELEASE_PROFILE}'" in config and "entrypoint: python3" in config
    assert "timeout: 7200s" in config and "machineType: E2_HIGHCPU_8" in config
    assert "diskSizeGb: '500'" in config and "logging: CLOUD_LOGGING_ONLY" in config
    assert "!profiles/*.json" in (HELPER.parent / ".gcloudignore").read_text()


@pytest.mark.parametrize("revision", ["main", "a" * 40])
def test_cli_requires_matching_revision_before_download(monkeypatch, revision):
    monkeypatch.setattr(build_context.subprocess, "run", lambda *a, **k: pytest.fail("no download"))
    with pytest.raises(SystemExit):
        build_context.main(["--profile", PROFILE_PATH.name, "--model-hf-revision", revision])


@pytest.mark.parametrize("filename", ["../escape.json", "/absolute.json", "x.json; false"])
def test_cli_rejects_profile_path_injection(filename):
    with pytest.raises(SystemExit):
        build_context.main(["--profile", filename, "--model-hf-revision", PROFILE.hub_revision])


def test_verify_only_checks_current_repository_without_download(monkeypatch):
    monkeypatch.setattr(build_context.subprocess, "run", lambda *a, **k: pytest.fail("no download"))
    assert build_context.main(["--profile", PROFILE_PATH.name, "--model-hf-revision", PROFILE.hub_revision, "--verify-only"]) == 0


@pytest.mark.parametrize("fail_download", [False, True])
def test_cli_uses_generation_pin_and_cleans_download(tmp_path, monkeypatch, fail_download):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    archive, profile = _archive(fixture)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copyfile(HELPER.parent / "Dockerfile", workspace / "Dockerfile")
    monkeypatch.setattr(build_context, "__file__", str(workspace / "prepare_context.py"))
    monkeypatch.setattr(build_context, "load_profile", lambda path: profile)
    calls = []

    def fake_download(args, *, check):
        calls.append((args, check))
        shutil.copyfile(archive, args[-1])
        if fail_download:
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(build_context.subprocess, "run", fake_download)
    args = ["--profile", "fixture.json", "--model-hf-revision", profile.hub_revision]
    if fail_download:
        with pytest.raises(subprocess.CalledProcessError):
            build_context.main(args)
    else:
        assert build_context.main(args) == 0
        assert (workspace / "build-context/Dockerfile").exists()
    assert calls == [(["gcloud", "storage", "cp", profile.candidate.gcs_uri, str(workspace / "candidate.tar")], True)]
    assert not (workspace / "candidate.tar").exists()


def test_cli_preserves_prior_artifacts_and_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(build_context, "__file__", str(tmp_path / "prepare_context.py"))
    monkeypatch.setattr(build_context, "load_profile", lambda path: PROFILE)
    monkeypatch.setattr(build_context.subprocess, "run", lambda *a, **k: pytest.fail("no download"))
    prior = tmp_path / "candidate.tar"
    prior.write_bytes(b"recovery data")
    with pytest.raises(FileExistsError):
        build_context.main(["--profile", PROFILE_PATH.name, "--model-hf-revision", PROFILE.hub_revision])
    assert prior.read_bytes() == b"recovery data"
