from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from edugraph_classify.dataset import DatasetConversionError, PromptTemplate, file_sha256


def test_tracked_prompt_loads_verbatim_with_stable_identity() -> None:
    path = Path(__file__).resolve().parents[1] / "prompts" / "direct-label-v3.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    prompt = PromptTemplate.load(path)

    assert prompt.to_mapping() == payload
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert prompt.sha256 == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("payload", [
    {"prompt_id": "direct", "version": "3", "system": "Classify"},
    {"prompt_id": "direct", "version": "3", "system": "Classify", "user": "Task", "extra": True},
])
def test_prompt_rejects_missing_or_extra_fields(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "prompt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetConversionError, match="unexpected fields"):
        PromptTemplate.load(path)


def test_file_sha256_hashes_complete_large_file_and_detects_changes(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    content = b"a" * (1024 * 1024 + 7)
    path.write_bytes(content)

    assert file_sha256(path) == hashlib.sha256(content).hexdigest()

    path.write_bytes(content[:-1] + b"b")
    assert file_sha256(path) != hashlib.sha256(content).hexdigest()
