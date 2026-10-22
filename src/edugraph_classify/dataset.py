"""Prompt identity and file hashes shared by RunPod preparation and recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


class DatasetConversionError(ValueError):
    """Raised when a tracked prompt violates the preparation contract."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    prompt_id: str
    version: str
    system: str
    user: str

    @classmethod
    def load(cls, path: Path) -> PromptTemplate:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != {"prompt_id", "version", "system", "user"}:
            raise DatasetConversionError("prompt file has unexpected fields")
        return cls(**payload)

    def to_mapping(self) -> dict[str, str]:
        """Use the tracked prompt verbatim across preparation and serving paths."""
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_mapping(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
