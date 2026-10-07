from __future__ import annotations

import pytest

from edugraph_classify.contracts import ContractError, LabelSet, canonicalize_structure


def test_label_set_parses_and_serializes_in_fixed_dimension_order() -> None:
    labels = LabelSet.from_mapping(
        {"abilities": ["Reason"], "areas": ["Addition"], "scopes": ["Within100"]}
    )

    assert labels.to_mapping() == {
        "areas": ["Addition"], "scopes": ["Within100"], "abilities": ["Reason"]
    }
    assert labels.to_json() == (
        '{"areas":["Addition"],"scopes":["Within100"],"abilities":["Reason"]}'
    )


def test_empty_dimensions_are_valid_and_keep_the_three_field_shape() -> None:
    assert LabelSet.from_mapping({"areas": [], "scopes": [], "abilities": []}).to_json() == (
        '{"areas":[],"scopes":[],"abilities":[]}'
    )


@pytest.mark.parametrize("payload", [
    {"areas": [], "scopes": []},
    {"areas": [], "scopes": [], "abilities": [], "other": []},
    {"areas": "Addition", "scopes": [], "abilities": []},
    {"areas": b"Addition", "scopes": [], "abilities": []},
    {"areas": None, "scopes": [], "abilities": []},
    {"areas": [""], "scopes": [], "abilities": []},
    {"areas": [" Addition"], "scopes": [], "abilities": []},
    {"areas": [1], "scopes": [], "abilities": []},
])
def test_label_set_rejects_invalid_shapes(payload: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        LabelSet.from_mapping(payload)


def test_canonicalization_reports_changes_without_inventing_or_pruning_labels() -> None:
    parsed = LabelSet(
        areas=("Square", "Rectangle", "Square"),
        scopes=("Within100",),
        abilities=(),
    )

    result = canonicalize_structure(parsed)

    assert parsed.areas == ("Square", "Rectangle", "Square")
    assert result.labels == LabelSet(areas=("Rectangle", "Square"), scopes=("Within100",))
    assert result.findings == (
        "removed duplicate areas labels",
        "sorted areas labels",
    )


def test_canonicalization_of_already_canonical_labels_has_no_findings() -> None:
    labels = LabelSet(areas=("Addition",), abilities=("Reason",))

    result = canonicalize_structure(labels)

    assert result.labels == labels
    assert result.findings == ()
