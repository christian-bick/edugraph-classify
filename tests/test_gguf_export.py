"""Offline checks for the pinned BF16 merge and GGUF export stages."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from edugraph_classify.gguf_export import (
    check_source, convert_gguf, export_gguf, file_sha256, merge_adapter, tree_sha256,
)


REPO = "Qwen/Qwen3.8-27B"
REVISION = "1" * 40
LLAMA_COMMIT = "2" * 40


def source_bundle(root: Path) -> Path:
    source = root / "selected"
    (source / "adapter").mkdir(parents=True)
    (source / "processor").mkdir()
    (source / "manifest.json").write_text(json.dumps({
        "run_id": "training-run", "recipe": {"model": {
            "hf_repository": REPO, "hf_revision": REVISION}}}), encoding="utf-8")
    (source / "prompt.json").write_text('{"system":"classify"}', encoding="utf-8")
    (source / "closed_schema.json").write_text('{"type":"object"}', encoding="utf-8")
    (source / "chat_template.jinja").write_text("{{ message }}", encoding="utf-8")
    (source / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (source / "processor" / "tokenizer.json").write_bytes(b"tokenizer")
    (source / "processor" / "config.json").write_text('{"old":true}', encoding="utf-8")
    return source


class FakeMerged:
    def save_pretrained(self, path, **kwargs):
        assert kwargs == {"safe_serialization": True, "max_shard_size": "5GB"}
        (path / "config.json").write_text('{"merged":true}', encoding="utf-8")
        (path / "model-00001-of-00001.safetensors").write_bytes(b"merged")


class FakeAdapter:
    def merge_and_unload(self, *, safe_merge):
        assert safe_merge is True
        return FakeMerged()


def fake_loaders(source: Path, seen: list):
    def load_model(repo, revision):
        seen.append(("model", repo, revision))
        return object()

    def load_adapter(base, path):
        seen.append(("adapter", path))
        assert path == source / "adapter"
        return FakeAdapter()

    return load_model, load_adapter


def fake_llama(root: Path) -> Path:
    llama = root / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    (llama / "convert_hf_to_gguf.py").write_text("# fake", encoding="utf-8")
    (llama / "build" / "bin" / "llama-quantize").write_text("fake", encoding="utf-8")
    return llama


def fake_commands(seen: list):
    def run(command, **kwargs):
        seen.append((command, kwargs))
        if command[0] == "git":
            return SimpleNamespace(stdout=LLAMA_COMMIT + "\n")
        if "--outfile" in command:
            output = Path(command[command.index("--outfile") + 1])
            output.write_bytes(b"vision" if "--mmproj" in command else b"language")
        else:
            Path(command[2]).write_bytes(b"quantized")
        return SimpleNamespace(stdout="")
    return run


def test_export_merges_from_pinned_bf16_base_and_converts_both_components(tmp_path):
    source = source_bundle(tmp_path)
    llama = fake_llama(tmp_path)
    loader_calls, commands = [], []
    model_loader, adapter_loader = fake_loaders(source, loader_calls)
    output = tmp_path / "gguf-output"

    result = export_gguf(source, output, model_repo=REPO, model_revision=REVISION,
                         llama_dir=llama, model_loader=model_loader,
                         adapter_loader=adapter_loader, command_runner=fake_commands(commands))

    assert loader_calls == [("model", REPO, REVISION), ("adapter", source / "adapter")]
    assert (output / "merged-hf" / "config.json").read_text(encoding="utf-8") == '{"merged":true}'
    assert (output / "merged-hf" / "tokenizer.json").read_bytes() == b"tokenizer"
    assert [command[0][0] for command in commands] == ["git", __import__("sys").executable,
                                                        __import__("sys").executable,
                                                        str(llama / "build" / "bin" / "llama-quantize")]
    assert commands[0][1] == {"check": True, "capture_output": True, "text": True}
    assert "--no-mtp" in commands[1][0] and "--mmproj" not in commands[1][0]
    assert "--mmproj" in commands[2][0]
    assert commands[3][0][-1] == "Q4_K_M"
    assert all(call[1] == {"cwd": llama, "check": True} for call in commands[1:])
    assert result["gguf"]["llama_cpp_commit"] == LLAMA_COMMIT
    assert result["gguf"]["language_quantized"] == {
        "path": str(output / "model-Q4_K_M.gguf"), "sha256": file_sha256(output / "model-Q4_K_M.gguf")}
    assert result["gguf"]["mmproj_bf16"]["sha256"] == file_sha256(output / "mmproj-BF16.gguf")
    assert result["merged_hf"]["files_sha256"] == tree_sha256(output / "merged-hf")
    assert result["source_files_sha256"] == result["contract_files_sha256"]
    assert json.loads((output / "export.json").read_text(encoding="utf-8")) == result


def test_invalid_source_or_existing_destination_stops_before_loading(tmp_path):
    source = source_bundle(tmp_path)
    llama = fake_llama(tmp_path)
    seen = []
    model_loader, adapter_loader = fake_loaders(source, seen)
    with pytest.raises(ValueError, match="disagree"):
        export_gguf(source, tmp_path / "out", model_repo=REPO, model_revision="3" * 40,
                    llama_dir=llama, model_loader=model_loader, adapter_loader=adapter_loader)
    with pytest.raises(ValueError, match="immutable"):
        check_source(source, REPO, "main")
    assert seen == []
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        export_gguf(source, tmp_path / "out", model_repo=REPO, model_revision=REVISION,
                    llama_dir=llama, model_loader=model_loader, adapter_loader=adapter_loader)


def test_missing_source_and_nested_destination_are_rejected(tmp_path):
    source = source_bundle(tmp_path)
    (source / "closed_schema.json").unlink()
    with pytest.raises(ValueError, match="closed_schema.json"):
        check_source(source, REPO, REVISION)
    (source / "closed_schema.json").write_text("{}", encoding="utf-8")
    (source / "adapter").rename(source / "removed-adapter")
    with pytest.raises(ValueError, match="adapter/"):
        check_source(source, REPO, REVISION)
    (source / "removed-adapter").rename(source / "adapter")
    with pytest.raises(ValueError, match="outside"):
        export_gguf(source, source / "nested", model_repo=REPO, model_revision=REVISION,
                    llama_dir=tmp_path / "unused")


def test_conversion_requires_pinned_tools_and_output_files(tmp_path):
    llama = fake_llama(tmp_path)
    merged = tmp_path / "merged"
    merged.mkdir()
    with pytest.raises(ValueError, match="quantization"):
        convert_gguf(merged, tmp_path / "out", llama_dir=llama, quantization="Q4; rm")
    (llama / "build" / "bin" / "llama-quantize").unlink()
    with pytest.raises(ValueError, match="converter and quantizer"):
        convert_gguf(merged, tmp_path / "out", llama_dir=llama)
    (llama / "build" / "bin" / "llama-quantize").write_text("fake", encoding="utf-8")

    def empty_run(command, **kwargs):
        return SimpleNamespace(stdout=LLAMA_COMMIT + "\n")

    with pytest.raises(ValueError, match="did not produce"):
        convert_gguf(merged, tmp_path / "out", llama_dir=llama, command_runner=empty_run)


def test_merge_can_be_reused_without_gguf_and_checks_llama_commit(tmp_path):
    source = source_bundle(tmp_path)
    llama = fake_llama(tmp_path)
    calls = []
    model_loader, adapter_loader = fake_loaders(source, calls)
    merged = merge_adapter(source, tmp_path / "merged", model_repo=REPO,
                           model_revision=REVISION, model_loader=model_loader,
                           adapter_loader=adapter_loader)
    assert merged["source_run_id"] == "training-run"
    assert merged["files_sha256"]["model-00001-of-00001.safetensors"] == file_sha256(
        tmp_path / "merged" / "model-00001-of-00001.safetensors")

    def wrong_commit(command, **kwargs):
        return SimpleNamespace(stdout="main\n")

    with pytest.raises(ValueError, match="resolve to a commit"):
        convert_gguf(tmp_path / "merged", tmp_path / "gguf", llama_dir=llama,
                     command_runner=wrong_commit)
