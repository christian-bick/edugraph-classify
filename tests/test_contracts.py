from __future__ import annotations

from dataclasses import replace

import pytest

from edugraph_classify.contracts import (
    ArtifactIdentity,
    ClassificationRequest,
    ContractError,
    ExecutionMode,
    LabelSet,
    ModelIdentity,
    PredictionViews,
    PromptSchemaIdentity,
    RawPrediction,
    RunIdentity,
    canonicalize_structure,
)


def test_label_set_parses_and_serializes_in_fixed_dimension_order() -> None:
    labels = LabelSet.from_mapping(
        {"abilities": ["Reason"], "areas": ["Addition"], "scopes": ["Within100"]}
    )

    assert labels.to_json() == (
        '{"areas":["Addition"],"scopes":["Within100"],"abilities":["Reason"]}'
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"areas": [], "scopes": []},
        {"areas": [], "scopes": [], "abilities": [], "other": []},
        {"areas": "Addition", "scopes": [], "abilities": []},
        {"areas": [""], "scopes": [], "abilities": []},
        {"areas": [" Addition"], "scopes": [], "abilities": []},
    ],
)
def test_label_set_rejects_noncanonical_shapes(payload: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        LabelSet.from_mapping(payload)


def test_structural_canonicalization_is_deterministic_and_visible() -> None:
    parsed = LabelSet(
        areas=("Subtraction", "Addition", "Addition"),
        scopes=("Within100",),
        abilities=("Reason",),
    )

    result = canonicalize_structure(parsed)

    assert result.labels.areas == ("Addition", "Subtraction")
    assert result.findings == (
        "removed duplicate areas labels",
        "sorted areas labels",
    )


def test_prediction_views_require_each_prior_stage() -> None:
    raw = RawPrediction(content="{}", metadata={"request_id": "provider-1"})
    parsed = LabelSet()

    with pytest.raises(ContractError, match="parsed"):
        PredictionViews(raw=raw, canonical=parsed)
    with pytest.raises(ContractError, match="canonical"):
        PredictionViews(raw=raw, parsed=parsed, derived=parsed)

    views = PredictionViews(raw=raw, parsed=parsed, canonical=parsed, derived=parsed)
    assert views.raw.metadata["request_id"] == "provider-1"


def test_raw_prediction_requires_text_content() -> None:
    with pytest.raises(ContractError, match="string"):
        RawPrediction(content=object())  # type: ignore[arg-type]


def test_run_and_request_identities_are_provider_neutral() -> None:
    run = RunIdentity(
        run_id="baseline-001",
        dataset=ArtifactIdentity("christian-bick/edugraph-exercises", "dataset-sha"),
        ontology=ArtifactIdentity("christian-bick/edugraph-ontology", "v0.21.0"),
        prompt_schema=PromptSchemaIdentity("direct", "1", "labels", "1"),
        code_commit="code-sha",
        model=ModelIdentity("fireworks", "accounts/fireworks/models/qwen3p5-9b"),
        execution_mode=ExecutionMode.PROVIDER_DEDICATED,
    )
    request = ClassificationRequest(
        request_id="validation/example.png",
        run_id=run.run_id,
        source_id="validation/example.png",
        image_sha256="A" * 64,
        solution=False,
    )

    assert run.model.provider == "fireworks"
    assert run.execution_mode is ExecutionMode.PROVIDER_DEDICATED
    assert request.image_sha256 == "a" * 64

    with pytest.raises(ContractError, match="digest"):
        replace(request, image_sha256="not-a-digest")
    with pytest.raises(ContractError, match="boolean"):
        replace(request, solution=0)  # type: ignore[arg-type]


def test_optional_identity_fields_are_validated() -> None:
    artifact = ArtifactIdentity("owner/data", "revision", content_hash="sha256:value")
    model = ModelIdentity("fireworks", "model", revision="provider-revision")

    assert artifact.content_hash == "sha256:value"
    assert model.revision == "provider-revision"
    with pytest.raises(ContractError, match="content_hash"):
        replace(artifact, content_hash=" ")
    with pytest.raises(ContractError, match="revision"):
        replace(model, revision=" ")


@pytest.mark.parametrize(
    ("factory", "arguments"),
    [
        (ArtifactIdentity, ("", "revision")),
        (ModelIdentity, ("fireworks", " model")),
        (PromptSchemaIdentity, ("direct", "1", "labels", "")),
    ],
)
def test_identities_reject_empty_or_padded_values(factory: object, arguments: tuple[str, ...]) -> None:
    with pytest.raises(ContractError):
        factory(*arguments)  # type: ignore[operator]
