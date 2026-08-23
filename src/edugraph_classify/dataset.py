"""Deterministic conversion of the pinned released dataset to VLM chat JSONL."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import fsspec
from PIL import Image as PillowImage

from .contracts import LabelSet
from .ontology import OntologyCatalog, OntologyError


class DatasetConversionError(ValueError):
    """Raised when released data violates the classifier ingestion contract."""


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

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceExample:
    split: str
    source_path: str
    tags: tuple[str, ...]
    solution: bool
    inline_image: bytes | None = None


@dataclass(frozen=True, slots=True)
class ConvertedExample:
    source: SourceExample
    image_sha256: str
    image_size: int
    mime_type: str
    labels: LabelSet
    provider_record: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SplitArtifact:
    split: str
    example_count: int
    questions: int
    solutions: int
    jsonl_path: str
    jsonl_sha256: str
    jsonl_size: int


@dataclass(frozen=True, slots=True)
class ConversionResult:
    splits: tuple[SplitArtifact, ...]
    source_manifest_path: str
    source_manifest_sha256: str
    profile_path: str
    profile_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "splits": [asdict(item) for item in self.splits],
            "source_manifest": {
                "path": self.source_manifest_path,
                "sha256": self.source_manifest_sha256,
            },
            "profile": {
                "path": self.profile_path,
                "sha256": self.profile_sha256,
            },
        }


_MIME_TYPES = {
    "BMP": "image/bmp",
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "PPM": "image/x-portable-pixmap",
    "TIFF": "image/tiff",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_example(
    row: Mapping[str, object],
    split: str,
    expected_uri_prefix: str,
) -> SourceExample:
    if set(row) != {"image", "tags", "solution"}:
        raise DatasetConversionError(f"{split} row has unexpected fields")
    image = row["image"]
    if not isinstance(image, Mapping) or set(image) != {"bytes", "path"}:
        raise DatasetConversionError(f"{split} row has an invalid image payload")
    source_path = image["path"]
    inline_image = image["bytes"]
    if not isinstance(source_path, str) or not source_path.startswith(expected_uri_prefix):
        raise DatasetConversionError(f"{split} row has an unpinned image URI")
    relative = source_path.removeprefix(expected_uri_prefix)
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts or path.parts[0] != split:
        raise DatasetConversionError(f"{split} row image URI escapes its official split")
    if inline_image is not None and not isinstance(inline_image, bytes):
        raise DatasetConversionError(f"{split} row image bytes are invalid")
    tags = row["tags"]
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise DatasetConversionError(f"{split} row tags must be an array of strings")
    solution = row["solution"]
    if not isinstance(solution, bool):
        raise DatasetConversionError(f"{split} row solution must be a boolean")
    return SourceExample(
        split=split,
        source_path=source_path,
        tags=tuple(tags),
        solution=solution,
        inline_image=inline_image,
    )


def _read_image(source: SourceExample) -> bytes:
    if source.inline_image is not None:
        return source.inline_image
    with fsspec.open(source.source_path, "rb").open() as stream:
        return stream.read()


def _image_metadata(content: bytes) -> tuple[str, str]:
    if not content:
        raise DatasetConversionError("dataset image is empty")
    encoded_size = ((len(content) + 2) // 3) * 4
    if encoded_size >= 10 * 1024 * 1024:
        raise DatasetConversionError("dataset image exceeds Fireworks' 10MB base64 input limit")
    try:
        with PillowImage.open(BytesIO(content)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
    except Exception as error:
        raise DatasetConversionError("dataset image cannot be decoded") from error
    if not isinstance(image_format, str) or image_format not in _MIME_TYPES:
        raise DatasetConversionError(f"unsupported dataset image format {image_format!r}")
    if width <= 0 or height <= 0:
        raise DatasetConversionError("dataset image has invalid dimensions")
    return _MIME_TYPES[image_format], hashlib.sha256(content).hexdigest()


def _provider_record(
    content: bytes,
    mime_type: str,
    labels: LabelSet,
    prompt: PromptTemplate,
) -> Mapping[str, object]:
    image_url = f"data:{mime_type};base64,{base64.b64encode(content).decode('ascii')}"
    return {
        "messages": [
            {"role": "system", "content": prompt.system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.user},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
            {"role": "assistant", "content": labels.to_json()},
        ]
    }


def _convert_one(
    source: SourceExample,
    catalog: OntologyCatalog,
    prompt: PromptTemplate,
    read_image: Callable[[SourceExample], bytes],
) -> ConvertedExample:
    try:
        labels = catalog.labels(list(source.tags))
    except OntologyError as error:
        raise DatasetConversionError(str(error)) from error
    content = read_image(source)
    mime_type, image_sha256 = _image_metadata(content)
    return ConvertedExample(
        source=source,
        image_sha256=image_sha256,
        image_size=len(content),
        mime_type=mime_type,
        labels=labels,
        provider_record=_provider_record(content, mime_type, labels, prompt),
    )


def _canonical_line(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _atomic_paths(paths: Iterable[Path]) -> dict[Path, Path]:
    return {path: path.with_suffix(path.suffix + ".tmp") for path in paths}


def convert_dataset(
    splits: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    repo_id: str,
    revision: str,
    catalog: OntologyCatalog,
    prompt: PromptTemplate,
    output_dir: Path,
    read_image: Callable[[SourceExample], bytes] = _read_image,
    workers: int = 8,
) -> ConversionResult:
    """Validate and serialize the two official splits without altering gold labels."""

    if set(splits) != {"train", "validation"}:
        raise DatasetConversionError("dataset must expose exactly train and validation splits")
    if workers < 1:
        raise DatasetConversionError("workers must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {split: output_dir / f"{split}.jsonl" for split in sorted(splits)}
    source_manifest = output_dir / "source-manifest.jsonl"
    profile_path = output_dir / "profile.json"
    final_paths = [*split_paths.values(), source_manifest, profile_path]
    temporary = _atomic_paths(final_paths)
    for path in temporary.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    expected_uri_prefix = f"hf://datasets/{repo_id}@{revision}/"
    seen_paths: set[str] = set()
    seen_hashes: dict[str, ConvertedExample] = {}
    duplicate_groups: dict[str, list[ConvertedExample]] = {}
    split_artifacts: list[SplitArtifact] = []
    profile: dict[str, Any] = {
        "splits": {},
        "label_frequencies": {"areas": {}, "scopes": {}, "abilities": {}},
    }

    try:
        with temporary[source_manifest].open("w", encoding="utf-8", newline="\n") as manifest:
            for split in sorted(splits):
                sources = sorted(
                    (
                        _source_example(row, split, expected_uri_prefix)
                        for row in splits[split]
                    ),
                    key=lambda item: item.source_path,
                )
                split_path = temporary[split_paths[split]]
                questions = 0
                solutions = 0
                cardinalities: Counter[int] = Counter()
                frequencies = {
                    "areas": Counter[str](),
                    "scopes": Counter[str](),
                    "abilities": Counter[str](),
                }
                count = 0
                with split_path.open("w", encoding="utf-8", newline="\n") as output:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        converted = pool.map(
                            lambda item: _convert_one(item, catalog, prompt, read_image),
                            sources,
                        )
                        for index, example in enumerate(converted):
                            if example.source.source_path in seen_paths:
                                raise DatasetConversionError(
                                    f"duplicate source path {example.source.source_path!r}"
                                )
                            seen_paths.add(example.source.source_path)
                            prior = seen_hashes.get(example.image_sha256)
                            if prior is not None:
                                if prior.source.split != example.source.split:
                                    raise DatasetConversionError(
                                        "cross-split duplicate image bytes for "
                                        f"{example.source.source_path!r} and "
                                        f"{prior.source.source_path!r}"
                                    )
                                duplicate_groups.setdefault(
                                    example.image_sha256, [prior]
                                ).append(example)
                            else:
                                seen_hashes[example.image_sha256] = example

                            output.write(_canonical_line(example.provider_record))
                            manifest.write(
                                _canonical_line(
                                    {
                                        "source_id": f"{split}:{index:06d}",
                                        "split": split,
                                        "source_path": example.source.source_path,
                                        "image_sha256": example.image_sha256,
                                        "image_size": example.image_size,
                                        "mime_type": example.mime_type,
                                        "solution": example.source.solution,
                                        "explicit_labels": example.labels.to_mapping(),
                                    }
                                )
                            )
                            count += 1
                            if example.source.solution:
                                solutions += 1
                            else:
                                questions += 1
                            cardinalities[
                                len(example.labels.areas)
                                + len(example.labels.scopes)
                                + len(example.labels.abilities)
                            ] += 1
                            for dimension in frequencies:
                                frequencies[dimension].update(getattr(example.labels, dimension))

                if count < 3:
                    raise DatasetConversionError(
                        f"{split} has {count} examples; Fireworks requires at least 3"
                    )
                profile["splits"][split] = {
                    "examples": count,
                    "questions": questions,
                    "solutions": solutions,
                    "label_cardinality": {
                        str(key): cardinalities[key] for key in sorted(cardinalities)
                    },
                }
                for dimension, counter in frequencies.items():
                    aggregate = profile["label_frequencies"][dimension]
                    for label, frequency in counter.items():
                        aggregate[label] = aggregate.get(label, 0) + frequency
                split_artifacts.append(
                    SplitArtifact(
                        split=split,
                        example_count=count,
                        questions=questions,
                        solutions=solutions,
                        jsonl_path=str(split_paths[split]),
                        jsonl_sha256=file_sha256(split_path),
                        jsonl_size=split_path.stat().st_size,
                    )
                )

        duplicate_details = []
        for image_sha256, examples in sorted(duplicate_groups.items()):
            label_variants = {example.labels.to_json() for example in examples}
            duplicate_details.append(
                {
                    "image_sha256": image_sha256,
                    "split": examples[0].source.split,
                    "source_paths": sorted(
                        example.source.source_path for example in examples
                    ),
                    "record_count": len(examples),
                    "conflicting_gold_labels": len(label_variants) > 1,
                }
            )
        profile["duplicate_images"] = {
            "within_split_groups": len(duplicate_details),
            "additional_records": sum(
                detail["record_count"] - 1 for detail in duplicate_details
            ),
            "conflicting_gold_label_groups": sum(
                bool(detail["conflicting_gold_labels"])
                for detail in duplicate_details
            ),
            "groups": duplicate_details,
        }

        temporary[profile_path].write_text(
            json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        source_sha = file_sha256(temporary[source_manifest])
        profile_sha = file_sha256(temporary[profile_path])
        for final, temp in temporary.items():
            os.replace(temp, final)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise

    return ConversionResult(
        splits=tuple(split_artifacts),
        source_manifest_path=str(source_manifest),
        source_manifest_sha256=source_sha,
        profile_path=str(profile_path),
        profile_sha256=profile_sha,
    )


def load_released_splits(
    repo_id: str,
    revision: str,
    *,
    cache_dir: Path,
) -> Mapping[str, Iterable[Mapping[str, object]]]:
    """Load the pinned public release directly through datasets in streaming mode."""

    from datasets import Image, load_dataset

    dataset = load_dataset(
        repo_id,
        revision=revision,
        streaming=True,
        cache_dir=str(cache_dir),
    )
    return {
        split: dataset[split].cast_column("image", Image(decode=False))
        for split in dataset
    }


def select_sorted_prefix(
    splits: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    limits: Mapping[str, object],
) -> Mapping[str, Iterable[Mapping[str, object]]]:
    """Select a deterministic metadata-only prefix for technical smoke runs."""

    if set(splits) != {"train", "validation"} or set(limits) != {
        "train",
        "validation",
    }:
        raise DatasetConversionError(
            "smoke selection requires train and validation splits and limits"
        )
    selected: dict[str, list[Mapping[str, object]]] = {}
    for split in ("train", "validation"):
        limit = limits[split]
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 3:
            raise DatasetConversionError(
                f"smoke {split} limit must be an integer of at least 3"
            )

        def source_path(row: Mapping[str, object]) -> str:
            image = row.get("image")
            if not isinstance(image, Mapping) or not isinstance(image.get("path"), str):
                raise DatasetConversionError(
                    f"smoke {split} row has an invalid image path"
                )
            return image["path"]

        rows = sorted(splits[split], key=source_path)
        if len(rows) < limit:
            raise DatasetConversionError(
                f"smoke {split} requested {limit} examples but only {len(rows)} exist"
            )
        selected[split] = rows[:limit]
    return selected
