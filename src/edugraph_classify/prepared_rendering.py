"""Recreate prepared vision examples and verify their pinned renderings."""

from __future__ import annotations

from pathlib import Path


def render_prepared(run: Path, config: dict, examples: list[dict], renderer) -> dict:
    rendered = {}
    for row in examples:
        example = renderer.render(
            (run / row["image_path"]).read_bytes(),
            row["target"],
            max_tokens=config["training"]["max_context_tokens"],
            max_output=config["evaluation"]["max_tokens"],
        )
        if example.fingerprint() != row["render_sha256"]:
            raise ValueError("rendering differs from the prepared manifest")
        rendered[row["id"]] = example
    return rendered
