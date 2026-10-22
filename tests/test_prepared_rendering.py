from __future__ import annotations

from dataclasses import dataclass

import pytest

from edugraph_classify.prepared_rendering import render_prepared


@dataclass
class Rendered:
    digest: str

    def fingerprint(self) -> str:
        return self.digest


class Renderer:
    def __init__(self, digest: str) -> None:
        self.digest = digest
        self.calls: list[tuple[bytes, str, int, int]] = []

    def render(self, image: bytes, target: str, *, max_tokens: int, max_output: int) -> Rendered:
        self.calls.append((image, target, max_tokens, max_output))
        return Rendered(self.digest)


def test_render_prepared_recreates_pinned_example(tmp_path) -> None:
    image = tmp_path / "images" / "task.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    renderer = Renderer("pinned-sha")
    config = {"training": {"max_context_tokens": 2048}, "evaluation": {"max_tokens": 512}}
    rows = [{"id": "task", "image_path": "images/task.png", "target": '{"areas":[]}',
             "render_sha256": "pinned-sha"}]

    rendered = render_prepared(tmp_path, config, rows, renderer)

    assert rendered == {"task": Rendered("pinned-sha")}
    assert renderer.calls == [(b"image", '{"areas":[]}', 2048, 512)]


def test_render_prepared_rejects_changed_rendering(tmp_path) -> None:
    image = tmp_path / "task.png"
    image.write_bytes(b"image")
    config = {"training": {"max_context_tokens": 2048}, "evaluation": {"max_tokens": 512}}
    rows = [{"id": "task", "image_path": "task.png", "target": "target",
             "render_sha256": "pinned-sha"}]

    with pytest.raises(ValueError, match="rendering differs from the prepared manifest"):
        render_prepared(tmp_path, config, rows, Renderer("changed-sha"))
