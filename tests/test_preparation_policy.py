from __future__ import annotations

import json
from copy import deepcopy

import pytest

from edugraph_classify import learning_data
from test_learning import ROOT, png, recipe, renderer_factory


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(learning_data, "audit_qwen_language_targets", lambda *args: {"verified": True})
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "config.json").write_text("{}")
    config = recipe()
    config["selection"].update(train="all", validation=2, final_validation="remaining")
    records = {
        split: [{"file_name": f"{i}.png", "labels": ["Subtraction" if split == "train" and i == 4 else "Addition"],
                 "solution": bool(i % 2)} for i in range(count)]
        for split, count in (("train", 5), ("validation", 7))
    }

    def prepare(name="run", *, selected=None):
        def fetch(url):
            split = "validation" if "/validation/" in url else "train"
            if url.endswith("metadata.jsonl"):
                return "\n".join(json.dumps(row) for row in records[split]).encode()
            i = int(url.rsplit("/", 1)[-1].split(".")[0])
            return png(i + (30 if split == "validation" else 0))
        run = tmp_path / name
        manifest = learning_data.prepare_learning(selected or config, ROOT, run, processor, "a" * 40,
            fetch, {}, renderer_factory, runtime_packages=("pillow",))
        return manifest, json.loads((run / "examples.json").read_text())

    return config, records, prepare


def test_all_and_remaining_preserve_odd_imbalanced_releases(release):
    config, records, prepare = release
    manifest, examples = prepare()
    assert manifest["cohort_counts"] == {"train": 5, "validation": 2, "final_validation": 5}
    assert manifest["resolved_selection"] == {"train": 5, "validation": 2, "final_validation": 5, "train_diagnostic": 2}
    assert manifest["unused_metadata_rows"] == {"train": 0, "validation": 0}
    assert len(examples) == sum(len(rows) for rows in records.values())
    assert manifest["optimizer_steps"] == 6  # Two epochs, ceil(5 / 2) updates each.
    assert manifest["recipe"]["selection"]["train"] == "all"
    assert manifest["recipe"]["selection"]["final_validation"] == "remaining"
    selected_ids = {row["id"] for row in examples if row["split"] == "validation"}
    final_ids = {row["id"] for row in examples if row["split"] == "final_validation"}
    assert not selected_ids & final_ids
    assert len(selected_ids | final_ids) == 7
    profile = manifest["data_profile"]
    assert profile["official_splits"]["train"]["views"] == {"question": 3, "solution": 2}
    assert profile["cohorts"]["train"]["label_support"] == {"Addition": 4, "Subtraction": 1}
    assert profile["cohorts"]["train"]["max_labels_by_dimension"] == {"areas": 1, "scopes": 0, "abilities": 0}
    assert profile["train_labels_missing_from_official_validation"] == ["Subtraction"]
    assert profile["train_labels_missing_from_selected_validation"] == ["Subtraction"]
    limits = manifest["token_limits"]
    assert limits["max_completion_tokens"] <= limits["configured_output_tokens"]
    assert limits["required_context_tokens"] == max(
        max(row["training_tokens"], row["prompt_tokens"] + config["evaluation"]["max_tokens"])
        for row in examples)
    generations = (1 + config["training"]["epochs"]) * (2 + 2) + 5
    assert manifest["token_budget"]["sample"] == generations * config["evaluation"]["max_tokens"]

    # A future larger release needs no count edit, and source order cannot change selection.
    for rows in records.values():
        rows.reverse()
    repeated, repeated_examples = prepare("reordered")
    assert repeated_examples == examples
    assert repeated["data_profile"] == profile
    records["train"].append({"file_name": "5.png", "labels": ["Addition"], "solution": True})
    expanded, _ = prepare("expanded")
    assert expanded["resolved_selection"]["train"] == 6


def test_numeric_limits_remain_explicit_balanced_subsets(release):
    config, _, prepare = release
    config["selection"].update(train=4, final_validation=2)
    manifest, examples = prepare()
    assert manifest["cohort_counts"] == {"train": 4, "validation": 2, "final_validation": 2}
    assert manifest["unused_metadata_rows"] == {"train": 1, "validation": 3}
    assert sum(row["solution"] for row in examples if row["split"] == "train") == 2


def test_remaining_final_and_all_training_cannot_silently_be_empty(release):
    config, records, prepare = release
    records["validation"] = records["validation"][:2]
    with pytest.raises(ValueError, match="selection.final_validation.*no examples remain"):
        prepare("empty-final")
    records["train"] = []
    with pytest.raises(ValueError, match="selection.train.*no examples remain"):
        prepare("empty-train")
    config["selection"]["final_validation"] = 0
    records["train"] = [{"file_name": "0.png", "labels": ["Addition"], "solution": False}]
    with pytest.raises(ValueError, match="not enough examples"):
        prepare("insufficient-diagnostic")


@pytest.mark.parametrize("key,value", [
    ("train", "remaining"), ("train", 0), ("train", True), ("train", 3),
    ("validation", "all"), ("validation", 0), ("validation", 3),
    ("final_validation", "all"), ("final_validation", -1), ("final_validation", False),
    ("train_diagnostic", "all"), ("train_diagnostic", -2),
])
def test_selection_policy_rejects_invalid_choices(key, value):
    config = recipe()
    config["selection"][key] = value
    with pytest.raises(ValueError):
        learning_data.validate_recipe(config, extra_fields=frozenset({"execution", "tracking", "artifacts"}))


def test_in_memory_recipe_validation_preserves_pins_and_historical_counts():
    config = recipe()
    extras = frozenset({"execution", "tracking", "artifacts"})
    assert learning_data.validate_recipe(config, extra_fields=extras) is config
    for update in ({"train": "all", "final_validation": "remaining"}, {"final_validation": 0}):
        changed = deepcopy(config)
        changed["selection"].update(update)
        assert learning_data.validate_recipe(changed, extra_fields=extras) is changed
    changed["dataset"]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        learning_data.validate_recipe(changed, extra_fields=extras)


@pytest.mark.parametrize("field,value,message", [
    ("max_context_tokens", 100, "example requires .*context tokens.*max_context_tokens=100"),
    ("max_tokens", 10, "gold completion.*evaluation.max_tokens=10"),
])
def test_preparation_errors_identify_the_source_example(release, field, value, message):
    config, _, prepare = release
    section = "training" if field == "max_context_tokens" else "evaluation"
    config[section][field] = value
    with pytest.raises(ValueError, match=r"train/\d+\.png: " + message):
        prepare()


def test_renderer_enforces_gold_output_and_reports_exact_context_requirement():
    renderer = renderer_factory()
    sample = renderer.render(png(), "{}", max_tokens=10000, max_output=512)
    count = len(sample.completion)
    with pytest.raises(ValueError, match=f"gold completion has {count} tokens"):
        renderer.render(png(), "{}", max_tokens=10000, max_output=count - 1)
    required = max(sample.training_tokens, sample.prompt_tokens + count)
    with pytest.raises(ValueError, match=f"requires {required} context tokens"):
        renderer.render(png(), "{}", max_tokens=required - 1, max_output=count)
    assert renderer.render(png(), "{}", max_tokens=required, max_output=count).training_tokens == sample.training_tokens
    renderer.image_processor.get_number_of_image_patches = lambda *a, **k: 0
    with pytest.raises(ValueError, match="at least one token"):
        renderer.render(png(), "{}", max_tokens=10000, max_output=512)
