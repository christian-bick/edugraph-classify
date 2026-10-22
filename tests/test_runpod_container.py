from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[1]
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"


def test_runpod_container_and_lock_pin_cuda_126_runtime() -> None:
    dockerfile = (ROOT / "containers" / "runpod-trainer" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert (
        "FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@"
        "sha256:23debbe74125dc84df96df79cff42079b3b15265c27140714fd27b5aa718faa4"
    ) in dockerfile
    assert "UV_NO_CACHE=1" in dockerfile
    assert "--no-install-project" in dockerfile
    assert "uv sync --frozen --no-dev --group training" in dockerfile
    assert 'ENTRYPOINT ["/app/.venv/bin/python", "-m", "edugraph_classify.runpod_worker"]' in dockerfile

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sources = project["tool"]["uv"]["sources"]
    assert sources["torch"] == sources["torchvision"] == [
        {"index": "pytorch-cpu", "marker": "sys_platform == 'win32'"},
        {"index": "pytorch-cu126", "marker": "sys_platform != 'win32'"},
    ]
    assert project["tool"]["uv"]["index"] == [
        {"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True},
        {"name": "pytorch-cu126", "url": CUDA_INDEX, "explicit": True}
    ]

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in lock["package"] if package["source"].get("registry") == CUDA_INDEX}
    assert packages["torch"]["version"] == "2.13.0+cu126"
    assert packages["torchvision"]["version"] == "0.28.0+cu126"
    assert packages["torch"]["source"] == packages["torchvision"]["source"] == {
        "registry": CUDA_INDEX
    }
