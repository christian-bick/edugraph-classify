"""Offline release checks use tiny stand-ins for the two large GGUF files."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from edugraph_classify import cli
from edugraph_classify.hf_gguf_release import (
    ARCHIVE_FILES, MAX_SMALL_FILE_BYTES, PUBLIC_FILES, SOURCE_FILES, load_release_recipe, prepare_release,
    run_hf_gguf_release_command,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def example(tmp_path: Path):
    files = {
        "model-Q4_K_M.gguf": b"tiny language weights",
        "mmproj-BF16.gguf": b"tiny vision projector",
        "chat_template.jinja": b"{{ messages }}",
        "prompt.json": b'{"user":"Classify this task image."}',
        "closed_schema.json": b'{"properties":{"abilities":{},"areas":{},"scopes":{}}}',
        "manifest.json": b'{"internal":"PRIVATE_OPERATIONAL_MARKER"}',
    }
    recipe = json.loads((Path(__file__).resolve().parents[1] / "experiments" /
                         "hf-gguf-release-qwen38-27b-v1.json").read_text(encoding="utf-8"))
    recipe["files_sha256"] = {name: sha(data) for name, data in files.items()}
    source = tmp_path / "source"
    source.mkdir()
    for name, data in files.items():
        (source / name).write_bytes(data)
    metadata = {
        "kind": "gguf_inference_candidate_v1", "run_id": recipe["candidate"]["run_id"],
        "source_training_run_id": recipe["candidate"]["source_training_run_id"],
        "source_model": {"sha256": recipe["candidate"]["source_model_sha256"],
                         "uri": "gs://PRIVATE_OPERATIONAL_MARKER"},
        "base_model": {"hf_repository": recipe["base_model"]["repository"],
                       "hf_revision": recipe["base_model"]["revision"],
                       "provider": "runpod"},
        "quantization": "Q4_K_M", "llama_cpp_commit": recipe["llama_cpp_commit"],
        "files_sha256": recipe["files_sha256"],
    }
    (source / "candidate.json").write_text(json.dumps(metadata), encoding="utf-8")
    archive = tmp_path / "candidate.tar"
    with tarfile.open(archive, "w") as tar:
        for name in sorted(ARCHIVE_FILES):
            tar.add(source / name, arcname=name)
    recipe["candidate"]["archive_bytes"] = archive.stat().st_size
    recipe["candidate"]["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, recipe, source, metadata


def test_prepare_stages_only_verified_public_files_and_marks_license_pending(tmp_path):
    archive, recipe, _, _ = example(tmp_path)
    output = tmp_path / "release"
    result = prepare_release(archive, recipe, output)
    assert result["license_gate"] == "blocked_owner_choice"
    assert set(path.name for path in output.iterdir()) == {
        *PUBLIC_FILES, "README.md", "provenance.json", "DO_NOT_PUBLISH_LICENSE_PENDING.txt"}
    assert result["files_sha256"] == {name: recipe["files_sha256"][name] for name in PUBLIC_FILES}
    assert not (output / "manifest.json").exists()
    assert not (output / "candidate.json").exists()
    assert "PRIVATE_OPERATIONAL_MARKER" not in "".join(
        path.read_text(encoding="utf-8") for path in output.iterdir() if path.suffix != ".gguf")
    card = (output / "README.md").read_text(encoding="utf-8")
    assert "DRAFT — DO NOT PUBLISH" in card and "license:" not in card
    assert "37/64" in card and "40/64" in card and "43/64" in card
    assert "181/250 final-assessment result belongs only to the NF4-plus-adapter" in card
    assert "areas`, `scopes`, `abilities` order" in card
    assert "--ubatch-size 1024" in card and "--mmproj mmproj-BF16.gguf" in card
    assert "library_name: gguf" in card
    assert f"blob/{recipe['benchmark_record_commit']}/docs/RUNPOD-GGUF-BENCHMARK" in card
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["candidate_archive_sha256"] == recipe["candidate"]["archive_sha256"]
    assert provenance["source_model_sha256"] == recipe["candidate"]["source_model_sha256"]
    assert provenance["model_license"] is None
    assert set(provenance["source_files_sha256"]) == set(SOURCE_FILES)


def test_license_requires_explicit_id_and_file_and_never_uploads(tmp_path):
    archive, recipe, _, _ = example(tmp_path)
    license_path = tmp_path / "chosen-license.txt"
    license_path.write_text("Owner-approved sample license", encoding="utf-8")
    recipe["model_license"]["upstream_license_sha256"] = sha(license_path.read_bytes())
    with pytest.raises(ValueError, match="together"):
        prepare_release(archive, recipe, tmp_path / "missing-file", license_id="apache-2.0")
    with pytest.raises(ValueError, match="together"):
        prepare_release(archive, recipe, tmp_path / "missing-id", license_file=license_path)
    with pytest.raises(ValueError, match="invalid model license"):
        prepare_release(archive, recipe, tmp_path / "bad-id", license_id="not valid", license_file=license_path)
    with pytest.raises(ValueError, match="missing or empty"):
        prepare_release(archive, recipe, tmp_path / "bad-file", license_id="apache-2.0",
                        license_file=tmp_path / "missing")
    with pytest.raises(ValueError, match="differs from release recipe"):
        prepare_release(archive, recipe, tmp_path / "wrong-id", license_id="mit", license_file=license_path)
    license_path.write_text("Different terms", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from pinned upstream LICENSE"):
        prepare_release(archive, recipe, tmp_path / "wrong-license", license_id="apache-2.0",
                        license_file=license_path)
    license_path.write_text("Owner-approved sample license", encoding="utf-8")
    output = tmp_path / "licensed"
    result = prepare_release(archive, recipe, output, license_id="apache-2.0", license_file=license_path)
    assert result["license_gate"] == "passed"
    assert (output / "LICENSE").read_text(encoding="utf-8") == "Owner-approved sample license"
    assert "license: apache-2.0" in (output / "README.md").read_text(encoding="utf-8")
    assert "License and attribution" in (output / "README.md").read_text(encoding="utf-8")
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["model_license_sha256"] == recipe["model_license"]["upstream_license_sha256"]
    assert not (output / "DO_NOT_PUBLISH_LICENSE_PENDING.txt").exists()
    with pytest.raises(FileExistsError):
        prepare_release(archive, recipe, output)


def test_archive_size_hash_and_candidate_metadata_fail_before_staging(tmp_path):
    archive, recipe, source, metadata = example(tmp_path)
    bad_size = json.loads(json.dumps(recipe))
    bad_size["candidate"]["archive_bytes"] += 1
    with pytest.raises(ValueError, match="archive differs"):
        prepare_release(archive, bad_size, tmp_path / "bad-size")
    bad_hash = json.loads(json.dumps(recipe))
    bad_hash["candidate"]["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="archive differs"):
        prepare_release(archive, bad_hash, tmp_path / "bad-hash")
    metadata["quantization"] = "Q8_0"
    (source / "candidate.json").write_text(json.dumps(metadata), encoding="utf-8")
    with tarfile.open(archive, "w") as tar:
        for name in sorted(ARCHIVE_FILES):
            tar.add(source / name, arcname=name)
    recipe["candidate"]["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="metadata differs"):
        prepare_release(archive, recipe, tmp_path / "bad-metadata")
    assert not (tmp_path / "bad-metadata").exists()


def test_member_hash_and_archive_members_are_checked(tmp_path):
    archive, recipe, source, _ = example(tmp_path)
    (source / "model-Q4_K_M.gguf").write_bytes(b"altered model")
    with tarfile.open(archive, "w") as tar:
        for name in sorted(ARCHIVE_FILES):
            tar.add(source / name, arcname=name)
    recipe["candidate"]["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="member checksum mismatch"):
        prepare_release(archive, recipe, tmp_path / "bad-member")
    assert not (tmp_path / "bad-member").exists()
    with tarfile.open(archive, "w") as tar:
        for name in sorted(ARCHIVE_FILES):
            tar.add(source / name, arcname=name)
        tar.add(source / "manifest.json", arcname="../escaped.json")
    recipe["candidate"]["archive_bytes"] = archive.stat().st_size
    recipe["candidate"]["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unsafe members"):
        prepare_release(archive, recipe, tmp_path / "bad-archive")


def test_small_contract_file_size_limit(tmp_path):
    archive, recipe, source, metadata = example(tmp_path)
    (source / "prompt.json").write_bytes(b"x" * (MAX_SMALL_FILE_BYTES + 1))
    recipe["files_sha256"]["prompt.json"] = sha((source / "prompt.json").read_bytes())
    metadata["files_sha256"] = recipe["files_sha256"]
    (source / "candidate.json").write_text(json.dumps(metadata), encoding="utf-8")
    with tarfile.open(archive, "w") as tar:
        for name in sorted(ARCHIVE_FILES):
            tar.add(source / name, arcname=name)
    recipe["candidate"]["archive_bytes"] = archive.stat().st_size
    recipe["candidate"]["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unexpectedly large"):
        prepare_release(archive, recipe, tmp_path / "large-prompt")
    assert not (tmp_path / "large-prompt").exists()


def test_recipe_pins_and_cli_output_boundary(tmp_path, monkeypatch):
    archive, recipe, _, _ = example(tmp_path)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    loaded = load_release_recipe(recipe_path)
    assert loaded == recipe
    args = cli.build_parser().parse_args(["hf-gguf-release", "--recipe", str(recipe_path),
                                          "--candidate-tar", str(archive), "--output", str(tmp_path / "cli-release")])
    status, result = run_hf_gguf_release_command(args)
    assert status == 0 and result["license_gate"] == "blocked_owner_choice"
    monkeypatch.chdir(tmp_path)
    args.output = str(tmp_path / "README.md")
    with pytest.raises(ValueError, match="ignored artifact"):
        run_hf_gguf_release_command(args)
    args.output = str(tmp_path)
    with pytest.raises(ValueError, match="ignored artifact"):
        run_hf_gguf_release_command(args)
    for mutate in (
        lambda r: r.update(repository="not/a/repo/name"),
        lambda r: r["candidate"].update(archive_sha256="main"),
        lambda r: r["candidate"].update(archive_bytes=0),
        lambda r: r["candidate"].update(run_id="not a run ID"),
        lambda r: r["base_model"].update(repository="invalid"),
        lambda r: r["base_model"].update(revision="main"),
        lambda r: r.update(benchmark_record_commit="main"),
        lambda r: r["model_license"].update(id="invalid license"),
        lambda r: r["model_license"].update(upstream_license_sha256="not a checksum"),
        lambda r: r["dataset"].update(repository="invalid"),
        lambda r: r.update(ontology_version="latest"),
        lambda r: r["files_sha256"].pop("manifest.json"),
        lambda r: r["evaluation"].update(gguf_exact=65),
    ):
        invalid = json.loads(json.dumps(recipe))
        mutate(invalid)
        recipe_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError):
            load_release_recipe(recipe_path)


def test_release_recipe_pins_exact_upstream_license():
    root = Path(__file__).resolve().parents[1]
    recipe = load_release_recipe(root / "experiments" / "hf-gguf-release-qwen38-27b-v1.json")
    upstream_license = root / "experiments" / "licenses" / "Qwen3.8-27B-1d4bf0f-LICENSE"
    assert upstream_license.is_file()
    assert sha(upstream_license.read_bytes()) == recipe["model_license"]["upstream_license_sha256"]
    assert recipe["model_license"]["id"] == "apache-2.0"
