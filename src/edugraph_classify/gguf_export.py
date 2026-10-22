"""Export a selected language adapter to a reproducible llama.cpp checkpoint.

The BF16 Hugging Face merge is a separate stage so another serving backend can
consume it without depending on GGUF conversion or llama.cpp.
"""
from __future__ import annotations

import gc
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


SOURCE_FILES = ("manifest.json", "prompt.json", "closed_schema.json", "chat_template.jinja")
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
QUANTIZATION = re.compile(r"[A-Z][A-Z0-9_]{1,31}\Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> dict[str, str]:
    """Hash every file in stable path order; the mapping identifies each shard."""
    return {path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def check_source(source_model_dir: Path, model_repo: str, model_revision: str) -> dict:
    if not SHA40.fullmatch(model_revision):
        raise ValueError("model revision must be an immutable 40-character commit")
    for name in SOURCE_FILES:
        if not (source_model_dir / name).is_file():
            raise ValueError(f"selected model bundle lacks {name}")
    for name in ("adapter", "processor"):
        if not (source_model_dir / name).is_dir():
            raise ValueError(f"selected model bundle lacks {name}/")
    manifest = json.loads((source_model_dir / "manifest.json").read_text(encoding="utf-8"))
    source_model = manifest["recipe"]["model"]
    if (source_model.get("hf_repository"), source_model.get("hf_revision")) != (model_repo, model_revision):
        raise ValueError("selected adapter and pinned base model disagree")
    return manifest


def _load_base(model_repo: str, model_revision: str):
    import torch
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        model_repo, revision=model_revision, dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=False, token=False,
    )


def _load_adapter(base, adapter_dir: Path):
    from peft import PeftModel

    return PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)


def merge_adapter(source_model_dir: Path, merged_dir: Path, *, model_repo: str,
                  model_revision: str, model_loader=None, adapter_loader=None) -> dict:
    """Merge the selected LoRA into the original BF16 base on host memory.

    This deliberately does not load the NF4 training configuration: a GGUF
    quantization must start from the merged high-precision weights.
    """
    source_model_dir, merged_dir = Path(source_model_dir).resolve(), Path(merged_dir).resolve()
    manifest = check_source(source_model_dir, model_repo, model_revision)
    merged_dir.mkdir(parents=True, exist_ok=False)
    base = (model_loader or _load_base)(model_repo, model_revision)
    adapter = (adapter_loader or _load_adapter)(base, source_model_dir / "adapter")
    merged = adapter.merge_and_unload(safe_merge=True)
    # The saved processor supplies the tokenizer and image preprocessing files
    # needed by the converter. The merged model writes its own config.json.
    shutil.copytree(source_model_dir / "processor", merged_dir, dirs_exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
    del base, adapter, merged
    gc.collect()
    return {"path": str(merged_dir), "model_repository": model_repo,
            "model_revision": model_revision, "source_run_id": manifest["run_id"],
            "files_sha256": tree_sha256(merged_dir)}


def convert_gguf(merged_dir: Path, output_dir: Path, *, llama_dir: Path,
                 quantization: str = "Q4_K_M", command_runner=None) -> dict:
    """Convert language and vision separately, quantizing language weights only."""
    if not QUANTIZATION.fullmatch(quantization):
        raise ValueError("invalid GGUF quantization name")
    merged_dir, output_dir, llama_dir = (
        Path(merged_dir).resolve(), Path(output_dir).resolve(), Path(llama_dir).resolve())
    converter = llama_dir / "convert_hf_to_gguf.py"
    quantizer = llama_dir / "build" / "bin" / "llama-quantize"
    if not converter.is_file() or not quantizer.is_file():
        raise ValueError("pinned llama.cpp converter and quantizer are required")
    run = command_runner or subprocess.run
    version = run(["git", "-C", str(llama_dir), "rev-parse", "HEAD"],
                  check=True, capture_output=True, text=True)
    llama_commit = version.stdout.strip()
    if not SHA40.fullmatch(llama_commit):
        raise ValueError("llama.cpp checkout must resolve to a commit")
    language_bf16 = output_dir / "model-BF16.gguf"
    projector_bf16 = output_dir / "mmproj-BF16.gguf"
    language_quantized = output_dir / f"model-{quantization}.gguf"
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = (
        [sys.executable, str(converter), str(merged_dir), "--outtype", "bf16",
         "--no-mtp", "--outfile", str(language_bf16)],
        [sys.executable, str(converter), str(merged_dir), "--outtype", "bf16",
         "--mmproj", "--outfile", str(projector_bf16)],
        [str(quantizer), str(language_bf16), str(language_quantized), quantization],
    )
    for command in commands:
        run(command, cwd=llama_dir, check=True)
    for path in (language_bf16, projector_bf16, language_quantized):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"GGUF export did not produce {path.name}")
    return {"llama_cpp_commit": llama_commit, "quantization": quantization,
            "language_bf16": {"path": str(language_bf16), "sha256": file_sha256(language_bf16)},
            "mmproj_bf16": {"path": str(projector_bf16), "sha256": file_sha256(projector_bf16)},
            "language_quantized": {"path": str(language_quantized), "sha256": file_sha256(language_quantized)}}


def export_gguf(source_model_dir: Path, output_dir: Path, *, model_repo: str,
                model_revision: str, llama_dir: Path, quantization: str = "Q4_K_M",
                model_loader=None, adapter_loader=None, command_runner=None) -> dict:
    """Build and hash an inference artifact from a verified training export.

    The caller owns downloading and verifying the immutable selected-model
    bundle. This function makes no provider mutation and preserves partial
    output for diagnosis when a stage fails.
    """
    source_model_dir, output_dir = Path(source_model_dir).resolve(), Path(output_dir).resolve()
    check_source(source_model_dir, model_repo, model_revision)
    if output_dir.exists():
        raise FileExistsError(f"GGUF export directory already exists: {output_dir}")
    if output_dir.resolve().is_relative_to(source_model_dir.resolve()):
        raise ValueError("GGUF output must be outside the selected model bundle")
    output_dir.mkdir(parents=True)
    merged = merge_adapter(source_model_dir, output_dir / "merged-hf",
                           model_repo=model_repo, model_revision=model_revision,
                           model_loader=model_loader, adapter_loader=adapter_loader)
    gguf = convert_gguf(output_dir / "merged-hf", output_dir, llama_dir=llama_dir,
                        quantization=quantization, command_runner=command_runner)
    for name in SOURCE_FILES:
        shutil.copy2(source_model_dir / name, output_dir / name)
    result = {"source_run_id": merged["source_run_id"], "source_model": {
        "repository": model_repo, "revision": model_revision},
        "source_files_sha256": {name: file_sha256(source_model_dir / name) for name in SOURCE_FILES},
        "adapter_files_sha256": tree_sha256(source_model_dir / "adapter"),
        "processor_files_sha256": tree_sha256(source_model_dir / "processor"),
        "merged_hf": merged, "gguf": gguf,
        "contract_files_sha256": {name: file_sha256(output_dir / name) for name in SOURCE_FILES}}
    (output_dir / "export.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result
