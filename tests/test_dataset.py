from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from edugraph_classify import dataset as dataset_module
from edugraph_classify.dataset import (
    DatasetConversionError,
    PromptTemplate,
    SourceExample,
    convert_dataset,
    select_sorted_prefix,
)
from edugraph_classify.ontology import OntologyCatalog


REPO = "owner/data"
REVISION = "a" * 40


def png(red: int) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (2, 2), (red, 0, 0)).save(stream, format="PNG")
    return stream.getvalue()


def row(split: str, index: int, color: int) -> dict[str, object]:
    return {
        "image": {
            "path": f"hf://datasets/{REPO}@{REVISION}/{split}/group/{index}.png",
            "bytes": png(color),
        },
        "tags": ["ProcedureUnderstanding", "Addition", "DegreeScale"],
        "solution": bool(index % 2),
    }


def prompt() -> PromptTemplate:
    return PromptTemplate("direct-label", "1", "system", "user")


def test_conversion_is_deterministic_and_keeps_dimension_aware_targets(tmp_path: Path) -> None:
    splits = {
        "train": [row("train", 2, 2), row("train", 0, 0), row("train", 1, 1)],
        "validation": [
            row("validation", 2, 5),
            row("validation", 0, 3),
            row("validation", 1, 4),
        ],
    }
    catalog = OntologyCatalog.load("0.21.0")

    first = convert_dataset(
        splits,
        repo_id=REPO,
        revision=REVISION,
        catalog=catalog,
        prompt=prompt(),
        output_dir=tmp_path / "one",
        workers=2,
    )
    second = convert_dataset(
        splits,
        repo_id=REPO,
        revision=REVISION,
        catalog=catalog,
        prompt=prompt(),
        output_dir=tmp_path / "two",
        workers=2,
    )

    assert [item.jsonl_sha256 for item in first.splits] == [
        item.jsonl_sha256 for item in second.splits
    ]
    assert first.source_manifest_sha256 == second.source_manifest_sha256
    train = next(item for item in first.splits if item.split == "train")
    assert train.example_count == 3
    assert train.questions == 2
    record = json.loads(Path(train.jsonl_path).read_text(encoding="utf-8").splitlines()[0])
    target = json.loads(record["messages"][-1]["content"])
    assert target == {
        "areas": ["Addition"],
        "scopes": ["DegreeScale"],
        "abilities": ["ProcedureUnderstanding"],
    }
    assert record["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_conversion_fails_closed_on_cross_split_duplicate_bytes(tmp_path: Path) -> None:
    duplicate = row("validation", 0, 1)
    splits = {
        "train": [row("train", 0, 1), row("train", 1, 2), row("train", 2, 3)],
        "validation": [duplicate, row("validation", 1, 4), row("validation", 2, 5)],
    }
    with pytest.raises(DatasetConversionError, match="duplicate image bytes"):
        convert_dataset(
            splits,
            repo_id=REPO,
            revision=REVISION,
            catalog=OntologyCatalog.load("0.21.0"),
            prompt=prompt(),
            output_dir=tmp_path / "duplicate",
        )


def test_conversion_preserves_and_reports_within_split_gold_conflicts(
    tmp_path: Path,
) -> None:
    conflicting = row("train", 1, 1)
    conflicting["tags"] = ["Addition", "ConceptClassification"]
    result = convert_dataset(
        {
            "train": [row("train", 0, 1), conflicting, row("train", 2, 2)],
            "validation": [
                row("validation", 0, 3),
                row("validation", 1, 4),
                row("validation", 2, 5),
            ],
        },
        repo_id=REPO,
        revision=REVISION,
        catalog=OntologyCatalog.load("0.21.0"),
        prompt=prompt(),
        output_dir=tmp_path / "within-split",
    )

    profile = json.loads(Path(result.profile_path).read_text(encoding="utf-8"))
    assert profile["duplicate_images"] == {
        "within_split_groups": 1,
        "additional_records": 1,
        "conflicting_gold_label_groups": 1,
        "groups": [
            {
                "image_sha256": profile["duplicate_images"]["groups"][0][
                    "image_sha256"
                ],
                "split": "train",
                "source_paths": sorted(
                    [
                        row("train", 0, 1)["image"]["path"],
                        conflicting["image"]["path"],
                    ]
                ),
                "record_count": 2,
                "conflicting_gold_labels": True,
            }
        ],
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extra": True}, "unexpected fields"),
        ({"image": "bad"}, "invalid image payload"),
        ({"image": {"path": f"hf://datasets/{REPO}@{REVISION}/train/../x.png", "bytes": b"x"}}, "escapes"),
        ({"image": {"path": f"hf://datasets/{REPO}@{REVISION}/train/x.png", "bytes": "bad"}}, "image bytes"),
        ({"tags": "Addition"}, "array of strings"),
        ({"solution": 1}, "solution must be a boolean"),
    ],
)
def test_source_row_validation_rejects_malformed_release_rows(
    change: dict[str, object], message: str
) -> None:
    payload = row("train", 0, 1)
    payload.update(change)
    with pytest.raises(DatasetConversionError, match=message):
        dataset_module._source_example(
            payload,
            "train",
            f"hf://datasets/{REPO}@{REVISION}/",
        )


def test_image_validation_and_remote_read_fail_closed(monkeypatch) -> None:
    with pytest.raises(DatasetConversionError, match="empty"):
        dataset_module._image_metadata(b"")
    with pytest.raises(DatasetConversionError, match="10MB"):
        dataset_module._image_metadata(b"x" * (10 * 1024 * 1024))
    with pytest.raises(DatasetConversionError, match="cannot be decoded"):
        dataset_module._image_metadata(b"not-an-image")

    class FakeImage:
        format = "WEBP"
        size = (1, 1)

        def __enter__(self) -> FakeImage:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def verify(self) -> None:
            return None

    monkeypatch.setattr(dataset_module.PillowImage, "open", lambda stream: FakeImage())
    with pytest.raises(DatasetConversionError, match="unsupported"):
        dataset_module._image_metadata(b"bytes")

    FakeImage.format = "PNG"
    FakeImage.size = (0, 1)
    with pytest.raises(DatasetConversionError, match="invalid dimensions"):
        dataset_module._image_metadata(b"bytes")

    monkeypatch.setattr(
        dataset_module.fsspec,
        "open",
        lambda *args, **kwargs: SimpleNamespace(open=lambda: BytesIO(b"remote")),
    )
    source = SourceExample("train", "hf://example", (), False)
    assert dataset_module._read_image(source) == b"remote"


def test_conversion_rejects_bad_split_shape_workers_labels_paths_and_counts(
    tmp_path: Path,
) -> None:
    catalog = OntologyCatalog.load("0.21.0")
    with pytest.raises(DatasetConversionError, match="exactly train"):
        convert_dataset(
            {"train": []},
            repo_id=REPO,
            revision=REVISION,
            catalog=catalog,
            prompt=prompt(),
            output_dir=tmp_path / "splits",
        )
    with pytest.raises(DatasetConversionError, match="workers"):
        convert_dataset(
            {"train": [], "validation": []},
            repo_id=REPO,
            revision=REVISION,
            catalog=catalog,
            prompt=prompt(),
            output_dir=tmp_path / "workers",
            workers=0,
        )

    unknown = row("train", 0, 1)
    unknown["tags"] = ["UnknownLabel"]
    with pytest.raises(DatasetConversionError, match="unknown ontology"):
        convert_dataset(
            {
                "train": [unknown, row("train", 1, 2), row("train", 2, 3)],
                "validation": [
                    row("validation", 0, 4),
                    row("validation", 1, 5),
                    row("validation", 2, 6),
                ],
            },
            repo_id=REPO,
            revision=REVISION,
            catalog=catalog,
            prompt=prompt(),
            output_dir=tmp_path / "unknown",
        )

    repeated_path = row("train", 0, 7)
    repeated_path["image"] = {
        "path": row("train", 1, 8)["image"]["path"],
        "bytes": png(7),
    }
    with pytest.raises(DatasetConversionError, match="duplicate source path"):
        convert_dataset(
            {
                "train": [repeated_path, row("train", 1, 8), row("train", 2, 9)],
                "validation": [
                    row("validation", 0, 10),
                    row("validation", 1, 11),
                    row("validation", 2, 12),
                ],
            },
            repo_id=REPO,
            revision=REVISION,
            catalog=catalog,
            prompt=prompt(),
            output_dir=tmp_path / "paths",
        )

    with pytest.raises(DatasetConversionError, match="at least 3"):
        convert_dataset(
            {
                "train": [row("train", 0, 13), row("train", 1, 14)],
                "validation": [
                    row("validation", 0, 15),
                    row("validation", 1, 16),
                    row("validation", 2, 17),
                ],
            },
            repo_id=REPO,
            revision=REVISION,
            catalog=catalog,
            prompt=prompt(),
            output_dir=tmp_path / "count",
        )


def test_prompt_and_dataset_loader_contracts(tmp_path: Path, monkeypatch) -> None:
    bad_prompt = tmp_path / "prompt.json"
    bad_prompt.write_text('{"prompt_id":"x"}', encoding="utf-8")
    with pytest.raises(DatasetConversionError, match="unexpected fields"):
        PromptTemplate.load(bad_prompt)
    assert len(prompt().sha256) == 64

    observed: dict[str, object] = {}

    class FakeSplit:
        def cast_column(self, name: str, image: object) -> str:
            observed[name] = image
            return "cast"

    module = ModuleType("datasets")
    module.Image = lambda **kwargs: kwargs  # type: ignore[attr-defined]

    def fake_load(*args: object, **kwargs: object) -> dict[str, FakeSplit]:
        observed["load"] = (args, kwargs)
        return {"train": FakeSplit(), "validation": FakeSplit()}

    module.load_dataset = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)

    result = dataset_module.load_released_splits(
        REPO, REVISION, cache_dir=tmp_path / "cache"
    )

    assert result == {"train": "cast", "validation": "cast"}
    assert observed["load"][1]["streaming"] is True

    escaped = row("train", 0, 10)
    escaped["image"] = {"path": "hf://datasets/other/repo@main/train/0.png", "bytes": png(10)}
    with pytest.raises(DatasetConversionError, match="unpinned"):
        convert_dataset(
            {
                "train": [escaped, row("train", 1, 11), row("train", 2, 12)],
                "validation": [
                    row("validation", 0, 13),
                    row("validation", 1, 14),
                    row("validation", 2, 15),
                ],
            },
            repo_id=REPO,
            revision=REVISION,
            catalog=OntologyCatalog.load("0.21.0"),
            prompt=prompt(),
            output_dir=tmp_path / "escaped",
        )


def test_smoke_selection_is_path_sorted_and_fail_closed() -> None:
    splits = {
        "train": [row("train", index, index) for index in (3, 1, 2, 0)],
        "validation": [
            row("validation", index, index + 10) for index in (3, 1, 2, 0)
        ],
    }
    selected = select_sorted_prefix(
        splits,
        limits={"train": 3, "validation": 3},
    )
    assert [item["image"]["path"] for item in selected["train"]] == [
        row("train", index, index)["image"]["path"] for index in (0, 1, 2)
    ]

    with pytest.raises(DatasetConversionError, match="at least 3"):
        select_sorted_prefix(splits, limits={"train": 2, "validation": 3})
    with pytest.raises(DatasetConversionError, match="only 4 exist"):
        select_sorted_prefix(splits, limits={"train": 5, "validation": 3})
    malformed = {**splits, "train": [{"image": None}, *splits["train"][1:]]}
    with pytest.raises(DatasetConversionError, match="invalid image path"):
        select_sorted_prefix(malformed, limits={"train": 3, "validation": 3})
