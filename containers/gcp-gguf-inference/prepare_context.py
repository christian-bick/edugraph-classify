"""Stage a reviewed, generation-pinned GGUF release for Cloud Build."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Mapping


IMAGE_FILES = ("model-Q4_K_M.gguf", "mmproj-BF16.gguf", "chat_template.jinja")
MEMBER_FILES = (*IMAGE_FILES, "prompt.json", "closed_schema.json", "manifest.json")
CHUNK_BYTES = 4 * 1024 * 1024
HUB_REPOSITORY = "christian-bick/EduGraph-Classifier-Qwen3.8-27B-GGUF"
RUNTIME_REPOSITORY = "europe-west4-docker.pkg.dev/edugraph-438718/edugraph-predict/qwen38-27b-gguf"


@dataclass(frozen=True)
class Candidate:
    gcs_uri: str
    archive_bytes: int
    archive_sha256: str
    member_sha256: Mapping[str, str]


@dataclass(frozen=True)
class Profile:
    candidate: Candidate
    hub_revision: str
    runtime_image: str
    llama_cpp_commit: str
    context_tokens: int
    batch_size: int
    ubatch_size: int
    evidence: Mapping[str, str]


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"invalid {label} fields")
    return value


def _match(value, pattern, label):
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError(f"invalid {label}")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON profile field")
        result[key] = value
    return result


def load_profile(path: Path) -> Profile:
    """Accept only inert release pins for the established 27B serving route."""
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    _keys(data, ("kind", "publication", "candidate", "runtime", "inference", "evidence"), "profile")
    if data["kind"] != "edugraph_gcp_gguf_build_profile_v1":
        raise ValueError("invalid profile kind")
    publication = _keys(data["publication"], ("repository", "revision"), "publication")
    if publication["repository"] != HUB_REPOSITORY:
        raise ValueError("unsupported Hub repository")
    revision = _match(publication["revision"], r"[0-9a-f]{40}", "Hub revision")
    candidate = _keys(data["candidate"], ("gcs_uri", "archive_bytes", "archive_sha256", "member_sha256"), "candidate")
    archive_hash = _match(candidate["archive_sha256"], r"[0-9a-f]{64}", "archive hash")
    uri = _match(candidate["gcs_uri"],
                 r"gs://edugraph-classify/runpod/inference-models/[a-z0-9][a-z0-9-]{0,127}/"
                 + archive_hash + r"\.tar#[1-9][0-9]{0,19}", "generation-pinned candidate URI")
    if type(candidate["archive_bytes"]) is not int or candidate["archive_bytes"] <= 0:
        raise ValueError("invalid archive byte count")
    hashes = _keys(candidate["member_sha256"], MEMBER_FILES, "member hashes")
    for value in hashes.values():
        _match(value, r"[0-9a-f]{64}", "member hash")
    runtime = _keys(data["runtime"], ("image", "llama_cpp_commit"), "runtime")
    image = _match(runtime["image"], re.escape(RUNTIME_REPOSITORY) + r"@sha256:[0-9a-f]{64}", "runtime image")
    commit = _match(runtime["llama_cpp_commit"], r"[0-9a-f]{40}", "llama.cpp commit")
    inference = _keys(data["inference"], ("context_tokens", "batch_size", "ubatch_size"), "inference")
    if any(type(value) is not int for value in inference.values()):
        raise ValueError("token settings must be integers")
    context, batch, ubatch = (inference[key] for key in ("context_tokens", "batch_size", "ubatch_size"))
    if context not in (2048, 4096) or not 1 <= ubatch <= batch <= context:
        raise ValueError("unsupported context or inconsistent batch sizes")
    evidence = _keys(data["evidence"], ("recipe_path", "recipe_sha256", "publication_path", "publication_sha256"), "evidence")
    for name in ("recipe", "publication"):
        _match(evidence[name + "_path"], r"experiments/[a-z0-9][a-z0-9-]*\.json", "evidence path")
        _match(evidence[name + "_sha256"], r"[0-9a-f]{64}", "evidence hash")
    return Profile(Candidate(uri, candidate["archive_bytes"], archive_hash, hashes),
                   revision, image, commit, context, batch, ubatch, evidence)


def verify_release_evidence(profile: Profile, repository: Path) -> None:
    """Bind a checked-in profile to its reviewed recipe and publication receipt offline."""
    evidence = {}
    for name in ("recipe", "publication"):
        path = repository / profile.evidence[name + "_path"]
        if sha256_file(path) != profile.evidence[name + "_sha256"]:
            raise ValueError(f"{name} evidence SHA-256 mismatch")
        evidence[name] = json.loads(path.read_text(encoding="utf-8"))
    recipe, publication = evidence["recipe"], evidence["publication"]
    candidate = profile.candidate
    if (recipe["repository"] != HUB_REPOSITORY or publication["repository"] != HUB_REPOSITORY
            or publication["revision"] != profile.hub_revision
            or publication["anonymous_verification"] != "passed"
            or recipe["base_model"]["repository"] != "Qwen/Qwen3.8-27B"
            or candidate.gcs_uri.split("/")[-2] != recipe["candidate"]["run_id"]
            or publication.get("candidate_run_id", recipe["candidate"]["run_id"]) != recipe["candidate"]["run_id"]
            or recipe["candidate"]["archive_sha256"] != candidate.archive_sha256
            or recipe["candidate"]["archive_bytes"] != candidate.archive_bytes
            or recipe["files_sha256"] != dict(candidate.member_sha256)
            or recipe["llama_cpp_commit"] != profile.llama_cpp_commit
            or recipe.get("context_tokens", 2048) != profile.context_tokens
            or recipe.get("ubatch_size", 1024) != profile.ubatch_size
            or max(2048, profile.ubatch_size) != profile.batch_size):
        raise ValueError("profile does not match verified release evidence")
    for name in (*IMAGE_FILES, "prompt.json", "closed_schema.json"):
        if publication["files"][name]["sha256"] != candidate.member_sha256[name]:
            raise ValueError("publication member hash mismatch")


def render_dockerfile(template: Path, profile: Profile) -> str:
    values = {
        "RUNTIME_IMAGE": profile.runtime_image,
        "LLAMA_CPP_COMMIT": profile.llama_cpp_commit,
        "LLAMA_CPP_SHORT_COMMIT": profile.llama_cpp_commit[:7],
        "MODEL_HF_REVISION": profile.hub_revision,
        "MODEL_SHA256": profile.candidate.member_sha256["model-Q4_K_M.gguf"],
        "PROJECTOR_SHA256": profile.candidate.member_sha256["mmproj-BF16.gguf"],
        "TEMPLATE_SHA256": profile.candidate.member_sha256["chat_template.jinja"],
        "CONTEXT_TOKENS": str(profile.context_tokens),
        "BATCH_SIZE": str(profile.batch_size),
        "UBATCH_SIZE": str(profile.ubatch_size),
    }
    source = template.read_text(encoding="utf-8")
    if set(re.findall(r"@@([A-Z_0-9]+)@@", source)) != set(values):
        raise ValueError("Dockerfile template fields do not match renderer")
    return re.sub(r"@@([A-Z_0-9]+)@@", lambda match: values[match[1]], source)


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


def prepare_context(archive_path: Path, output_dir: Path, *, profile: Profile,
                    dockerfile: Path | None = None) -> None:
    """Verify the whole archive and every source member before staging three files."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    candidate = profile.candidate
    if archive_path.stat().st_size != candidate.archive_bytes:
        raise ValueError("candidate archive byte count mismatch")
    if sha256_file(archive_path) != candidate.archive_sha256:
        raise ValueError("candidate archive SHA-256 mismatch")
    rendered = render_dockerfile(dockerfile or Path(__file__).with_name("Dockerfile"), profile)
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
        (stage / "Dockerfile").write_text(rendered, encoding="utf-8", newline="\n")
        os.replace(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Reviewed JSON filename under profiles/")
    parser.add_argument("--model-hf-revision", required=True)
    parser.add_argument("--verify-only", action="store_true", help="Check repository evidence without downloading")
    args = parser.parse_args(argv)
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*\.json", args.profile) is None:
        parser.error("--profile must be a JSON filename under profiles/")
    workspace = Path(__file__).resolve().parent
    profile = load_profile(workspace / "profiles" / args.profile)
    if args.model_hf_revision != profile.hub_revision:
        parser.error("--model-hf-revision must match the reviewed profile")
    if args.verify_only:
        verify_release_evidence(profile, workspace.parents[1])
        render_dockerfile(workspace / "Dockerfile", profile)
        return 0
    archive_path, output_dir = workspace / "candidate.tar", workspace / "build-context"
    if archive_path.exists() or output_dir.exists():
        raise FileExistsError("Cloud Build workspace must not contain a prior candidate or context")
    try:
        subprocess.run(["gcloud", "storage", "cp", profile.candidate.gcs_uri, str(archive_path)], check=True)
        prepare_context(archive_path, output_dir, profile=profile)
    finally:
        archive_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
