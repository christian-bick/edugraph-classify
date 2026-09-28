"""Check that pinned Qwen LoRA targets are confined to the language model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

LANGUAGE_PREFIX = "model.language_model."


def assess_language_targets(linear_modules: list[str], targets: list[str]) -> dict:
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("expected LoRA targets must be a nonempty unique list")
    names = [name for name in linear_modules if name.rsplit(".", 1)[-1] in targets]
    matched = {name.rsplit(".", 1)[-1] for name in names}
    if matched != set(targets) or any(not name.startswith(LANGUAGE_PREFIX) for name in names):
        raise ValueError("LoRA targets are missing or include non-language modules")
    paths = sorted(names)
    return {"expected_target_modules": sorted(targets), "language_linear_modules": len(paths),
            "matched_module_paths_sha256": hashlib.sha256(
                json.dumps(paths, separators=(",", ":")).encode()
            ).hexdigest(), "vision_or_bridge_modules": 0}


def audit_qwen_language_targets(processor: Path, targets: list[str]) -> dict:
    """Resolve module names on meta tensors; no weights or GPU are loaded."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(processor, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    linear_modules = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    return assess_language_targets(linear_modules, targets)
