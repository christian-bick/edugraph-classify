from __future__ import annotations

import io
import json
import logging
import random
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from jinja2 import Environment
from PIL import Image

from edugraph_classify import cli, learning_cli, learning_data, learning_run
from edugraph_classify.dataset import PromptTemplate, convert_dataset
from edugraph_classify.evaluation import evaluate, parse_prediction
from edugraph_classify.ontology import OntologyCatalog
from edugraph_classify.output_contract import closed_vocabulary_schema
from edugraph_classify.learning_resume import verify_resume_source
from edugraph_classify.providers.fireworks_training_api import FireworksTrainingAPI, inspect_training_model, to_datum
from edugraph_classify.qwen_training_policy import assess_language_targets
from edugraph_classify.qwen_rendering import IMAGE_MARKER, QwenVisionRenderer, VisionExample, fixed_system_template

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "experiments/edugraph-20260928-qwen38-27b-learning-v1.json"
GOLD = {"areas": ["Addition"], "scopes": [], "abilities": []}


class Tokenizer:
    chat_template = "{%- for m in messages %}{{ '<|im_start|>' + m.role + '\n' }}{{ m.content if m.content is string else m.content[0].text + '<|vision_start|><|image_pad|><|vision_end|>' }}{{ '<|im_end|>\n' }}{%- endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def encode(self, text, **kwargs):
        tokens = []
        parts = text.split(IMAGE_MARKER)
        for index, part in enumerate(parts):
            if index:
                tokens.append(1_000_000)
            tokens.extend(map(ord, part))
        return tokens

    def apply_chat_template(self, messages, *, chat_template=None, **kwargs):
        def fail(message):
            raise ValueError(message)
        return Environment().from_string(chat_template or self.chat_template).render(messages=messages, raise_exception=fail, **kwargs)


def renderer_factory(path=None, prompt=None):
    prompt = prompt or {"system": "fixed system", "user": "classify image"}
    return QwenVisionRenderer(Tokenizer(), NS(merge_size=2, get_number_of_image_patches=lambda *a, **k: 28), prompt["system"], prompt["user"])


def png(color=0):
    output = io.BytesIO()
    Image.new("RGB", (5, 6), (color, 0, 0)).save(output, format="PNG")
    return output.getvalue()


def recipe():
    value = learning_data.load_recipe(RECIPE)
    value["prompt_path"] = "prompts/direct-label-v3.json"
    value["selection"] = {"train": 4, "validation": 4, "train_diagnostic": 2, "seed": 42}
    value["training"].update(epochs=2, batch_size=2, max_context_tokens=50000)
    return value


@pytest.fixture
def prepared(tmp_path):
    processor = tmp_path / "original-processor"
    processor.mkdir()
    (processor / "config.json").write_text("{}")
    (processor / ".ignored").mkdir()
    (processor / ".hidden").write_text("skip")
    records = [{"file_name": f"{i}.png", "labels": ["Addition"], "solution": bool(i % 2)} for i in range(4)]
    def fetch(url):
        if url.endswith("metadata.jsonl"):
            return "\n".join(json.dumps(r) for r in records).encode()
        index = int(url.rsplit("/", 1)[-1].split(".")[0])
        return png(index + (10 if "/validation/" in url else 0))
    run = tmp_path / "run"
    manifest = learning_data.prepare_learning(recipe(), ROOT, run, processor, "a" * 40, fetch, {"read_only": True}, renderer_factory)
    return NS(run=run, manifest=manifest, processor=processor, fetch=fetch, path=run / "manifest.json")


def test_assistant_only_loss_and_one_image_survive_target_shift():
    renderer = renderer_factory()
    example = renderer.render(png(), json.dumps(GOLD), max_tokens=10000, max_output=20)
    targets, weights = example.targets_and_weights()
    assert example.image_tokens == 7
    assert len(targets) == len(weights) == example.training_tokens
    assert all(w == 0 for w in weights[:example.prompt_tokens - 1])
    assert sum(weights) == len(example.completion)
    assert "".join(chr(t) for t, w in zip(targets, weights) if w) == json.dumps(GOLD) + "<|im_end|>\n"
    assert example.sampling_tokens().count(1_000_000) == 1
    assert len(example.fingerprint()) == 64
    assert example.fingerprint() != renderer.render(png(250), json.dumps(GOLD), max_tokens=10000, max_output=20).fingerprint()
    with pytest.raises(ValueError, match="truncation"):
        renderer.render(png(), json.dumps(GOLD), max_tokens=10, max_output=20)


def test_embedded_system_template_matches_and_rejects_overrides():
    renderer = renderer_factory()
    template = renderer.export_template()
    expected = renderer.tokenizer.apply_chat_template(renderer.messages(), add_generation_prompt=True)
    assert renderer.tokenizer.apply_chat_template(renderer.messages(), chat_template=template, add_generation_prompt=True) == expected
    bad = [{"role": "system", "content": "override"}, *renderer.messages()[1:]]
    with pytest.raises(ValueError, match="fixed"):
        renderer.tokenizer.apply_chat_template(bad, chat_template=template)
    assert "enable_thinking = false" in fixed_system_template("x", "system")


def test_rendering_fails_closed_on_template_and_marker_changes(monkeypatch):
    tokenizer = Tokenizer()
    tokenizer.encode = lambda *a, **k: [1, 2]
    with pytest.raises(ValueError, match="one special token"):
        QwenVisionRenderer(tokenizer, None, "x", "y")
    renderer = renderer_factory()
    monkeypatch.setattr(renderer.tokenizer, "apply_chat_template", lambda messages, **kw: "different" if len(messages) == 3 else "prefix")
    with pytest.raises(ValueError, match="prefixes"):
        renderer.render(png(), "{}", max_tokens=10000, max_output=20)
    renderer = renderer_factory()
    renderer.user = IMAGE_MARKER
    with pytest.raises(ValueError, match="one independent"):
        renderer.render(png(), "{}", max_tokens=10000, max_output=20)
    renderer = renderer_factory()
    original = renderer.tokenizer.apply_chat_template
    monkeypatch.setattr(renderer.tokenizer, "apply_chat_template", lambda *a, **kw: "bad" if "chat_template" in kw else original(*a, **kw))
    with pytest.raises(ValueError, match="differs"):
        renderer.export_template()


def test_tokenizer_load_uses_released_backend_without_rebuilding(tmp_path, monkeypatch):
    (tmp_path / "tokenizer_config.json").write_text('{"add_bos_token":false}')
    tokenizer_loader, image_loader = Mock(return_value=Tokenizer()), Mock(return_value=NS())
    monkeypatch.setitem(sys.modules, "transformers", NS(TokenizersBackend=NS(from_pretrained=tokenizer_loader), AutoImageProcessor=NS(from_pretrained=image_loader)))
    QwenVisionRenderer.load(tmp_path, {"system": "x", "user": "y"})
    assert tokenizer_loader.call_args.kwargs == {"local_files_only": True, "add_bos_token": False}


@pytest.mark.parametrize("path", ["../escape", "/root", "C:/drive", "x\\y", ""])
def test_paths_are_contained(path):
    with pytest.raises(ValueError, match="safe relative"):
        learning_data.safe_relative(path)


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(unknown=True), lambda c: c.update(run_id="Bad/Name"),
    lambda c: c["dataset"].update(revision="main"), lambda c: c["dataset"].update(label_field="tags"),
    lambda c: c["training"].update(epochs=0), lambda c: c["selection"].update(train=3),
    lambda c: c["selection"].update(train_diagnostic=500), lambda c: c["training"].update(learning_rate=float("nan")),
    lambda c: c["evaluation"].update(temperature=1),
])
def test_recipe_rejects_unsafe_or_unpinned_runs(tmp_path, mutate):
    config = recipe()
    mutate(config)
    path = tmp_path / "recipe.json"
    learning_data.write_json(path, config)
    with pytest.raises(ValueError):
        learning_data.load_recipe(path)


def test_epoch_evaluation_requires_valid_patience(tmp_path):
    for value in (0, -1, 1.5, True):
        config = recipe()
        config["training"]["early_stop_patience"] = value
        path = tmp_path / "recipe.json"
        learning_data.write_json(path, config)
        with pytest.raises(ValueError, match="early_stop_patience"):
            learning_data.load_recipe(path)
    config["evaluation"]["every_epoch"] = True
    config["training"]["early_stop_patience"] = 1
    learning_data.write_json(path, config)
    assert learning_data.load_recipe(path)["training"]["early_stop_patience"] == 1


def test_metadata_and_balanced_selection_do_not_depend_on_input_order():
    catalog = OntologyCatalog.load("0.30.0")
    rows = [{"file_name": f"{i}.png", "labels": ["Addition"], "solution": bool(i % 2)} for i in range(20)]
    assert learning_data.select_cohort(rows, 6, 42) == learning_data.select_cohort(list(reversed(rows)), 6, 42)
    assert sum(r["solution"] for r in learning_data.select_cohort(rows, 6, 42)) == 3
    for invalid in ([{**rows[0], "extra": 1}], [rows[0], rows[0]], [{**rows[0], "solution": 1}]):
        with pytest.raises(ValueError):
            learning_data.metadata_rows("\n".join(json.dumps(r) for r in invalid).encode(), catalog)
    with pytest.raises(ValueError, match="not enough"):
        learning_data.select_cohort(rows[:1], 4, 42)


def test_final_validation_is_disjoint_and_counted_once(prepared):
    config = recipe()
    config["run_id"] = "final-test"
    config["selection"].update(validation=2, final_validation=2)
    run = prepared.run.parent / "final-run"
    manifest = learning_data.prepare_learning(config, ROOT, run, prepared.processor, "a" * 40,
        prepared.fetch, {"read_only": True}, renderer_factory)
    assert manifest["cohort_counts"] == {"train": 4, "validation": 2, "final_validation": 2}
    _, rows = learning_data.verify_prepared(run / "manifest.json", "a" * 40)
    selected = {r["id"] for r in rows if r["split"] == "validation"}
    final = {r["id"] for r in rows if r["split"] == "final_validation"}
    assert len(selected | final) == 4 and not selected & final
    assert manifest["token_budget"]["sample"] == (2 * 4 + 2) * config["evaluation"]["max_tokens"]
    cohort = NS(run=run, manifest=manifest, path=run / "manifest.json")
    result = execute(cohort, FakeAdapter())
    assert result["status"] == "completed"
    comparison = json.loads((run / "comparison.json").read_text())
    assert comparison["final_validation"]["n"] == 2
    assert len(json.loads((run / "final-predictions.json").read_text())) == 2


def test_conflicting_identical_train_images_block_provider_start(prepared):
    config = recipe()
    config["run_id"] = "conflict-test"
    run = prepared.run.parent / "conflict-run"
    rows = [{"file_name": f"{i}.png", "labels": ["Subtraction" if i == 1 else "Addition"],
             "solution": bool(i % 2)} for i in range(4)]
    def fetch(url):
        if url.endswith("metadata.jsonl"):
            return "\n".join(json.dumps(row) for row in rows).encode()
        index = int(url.rsplit("/", 1)[-1].split(".")[0])
        return png(0 if "/train/" in url and index == 1 else index + (10 if "/validation/" in url else 0))
    manifest = learning_data.prepare_learning(config, ROOT, run, prepared.processor, "a" * 40,
        fetch, {"read_only": True}, renderer_factory)
    assert manifest["duplicate_image_counts"]["train"] == 1
    assert len(manifest["training_data_conflicts"]) == 1
    assert {row["gold"]["areas"][0] for row in manifest["training_data_conflicts"][0]["examples"]} == {"Addition", "Subtraction"}
    factory = Mock()
    with pytest.raises(ValueError, match="conflicting gold"):
        learning_run.execute_learning(run / "manifest.json", config["run_id"], "a" * 40,
            factory, Mock(), renderer_factory=renderer_factory)
    factory.assert_not_called()
    assert not (run / "execution.json").exists()


def test_preparation_pins_audits_and_detects_tampering(prepared):
    manifest, examples = learning_data.verify_prepared(prepared.path, "a" * 40)
    assert manifest["metadata_rows_validated"] == {"train": 4, "validation": 4}
    assert manifest["selected_images_validated"] == 8
    assert manifest["optimizer_steps"] == 4
    assert len([r for r in examples if r["train_diagnostic"]]) == 2
    assert manifest["estimated_max_token_cost_usd"] > 0
    prompt = json.loads((prepared.run / "prompt.json").read_text())
    assert prompt == PromptTemplate.load(ROOT / "prompts/direct-label-v3.json").to_mapping()
    assert "Allowed ontology vocabulary" not in prompt["system"]
    closed = json.loads((prepared.run / "closed_schema.json").read_text())
    assert "Addition" in closed["properties"]["areas"]["items"]["enum"]
    assert "Addition" not in closed["properties"]["abilities"]["items"]["enum"]
    assert not (prepared.run / "processor/.hidden").exists()
    with pytest.raises(ValueError, match="same clean code"):
        learning_data.verify_prepared(prepared.path, "b" * 40)
    (prepared.run / "prompt.json").write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        learning_data.verify_prepared(prepared.path, "a" * 40)


def test_compact_prompt_matches_shared_jsonl_conversion_and_embedded_template(prepared, tmp_path):
    prompt = PromptTemplate.load(ROOT / "prompts/direct-label-v3.json")
    revision = "b" * 40
    splits = {split: [{"image": {"bytes": png(index * 10 + i), "path": f"hf://datasets/owner/data@{revision}/{split}/{i}.png"},
                       "labels": ["Addition"], "solution": False} for i in range(3)]
              for index, split in enumerate(("train", "validation"))}
    converted = convert_dataset(splits, repo_id="owner/data", revision=revision,
        catalog=OntologyCatalog.load("0.30.0"), prompt=prompt, output_dir=tmp_path / "jsonl", label_field="labels")
    for artifact in converted.splits:
        for line in Path(artifact.jsonl_path).read_text().splitlines():
            record = json.loads(line)
            assert record["messages"][0] == {"role": "system", "content": prompt.system}
            assert record["messages"][1]["content"][0]["text"] == prompt.user
    renderer = renderer_factory(prompt=prompt.to_mapping())
    template = (prepared.run / "chat_template.jinja").read_text()
    assert renderer.tokenizer.apply_chat_template(renderer.messages()[1:], chat_template=template,
        add_generation_prompt=True) == renderer.tokenizer.apply_chat_template(renderer.messages(), add_generation_prompt=True)


def test_closed_schema_and_language_only_targets_fail_closed():
    catalog = OntologyCatalog.load("0.30.0")
    schema = json.loads((ROOT / "schemas/dimension-labels-v1.json").read_text())
    closed = closed_vocabulary_schema(schema, catalog)
    assert closed["required"] == ["areas", "scopes", "abilities"]
    assert closed["properties"]["areas"]["items"]["enum"] == sorted(closed["properties"]["areas"]["items"]["enum"])
    assert "enum" not in schema["properties"]["areas"]["items"]
    with pytest.raises(ValueError, match="exactly"):
        closed_vocabulary_schema({"properties": {"areas": {}}}, catalog)
    names = ["model.language_model.layers.0.mlp.down_proj", "model.visual.blocks.0.attn.qkv"]
    assert assess_language_targets(names, ["down_proj"])["vision_or_bridge_modules"] == 0
    for targets in (["qkv"], ["missing"], ["down_proj", "down_proj"], []):
        with pytest.raises(ValueError):
            assess_language_targets(names, targets)


@pytest.mark.parametrize("failure", ["overlap", "cost", "large"])
def test_preparation_rejects_leakage_cost_and_large_images(prepared, failure):
    config = recipe()
    config["estimated_cost_limit_usd"] = 1e-10 if failure == "cost" else 20
    def fetch(url):
        if not url.endswith("metadata.jsonl"):
            if failure == "overlap":
                return png()
            if failure == "large":
                return b"x" * 7_500_001
        return prepared.fetch(url)
    with pytest.raises(ValueError):
        learning_data.prepare_learning(config, ROOT, prepared.run.parent / "bad", prepared.processor, "a" * 40, fetch, {}, renderer_factory)


def test_evaluation_counts_raw_errors_without_ontology_repair():
    catalog = OntologyCatalog.load("0.30.0")
    assert not parse_prediction("```json\n{}\n```", catalog)["valid"]
    assert not parse_prediction("[]", catalog)["valid"]
    duplicate = parse_prediction('{"areas":["Addition","Addition"],"scopes":[],"abilities":[]}', catalog)
    assert not duplicate["valid"] and duplicate["canonical_valid"]
    examples = [{"id": str(i), "gold": GOLD, "solution": bool(i % 2)} for i in range(4)]
    texts = [json.dumps(GOLD), "garbage", '{"areas":["Invented"],"scopes":[],"abilities":[]}', '{"areas":["Addition","Addition"],"scopes":[],"abilities":[]}']
    predictions = [{"id": str(i), "text": text, "latency_seconds": i + 1, "usage": {"prompt_tokens": 20, "completion_tokens": 4}} for i, text in enumerate(texts)]
    report = evaluate(examples, predictions, examples, catalog)
    assert report["explicit"]["exact_set_match"] == .25
    assert report["canonical"]["exact_set_match"] == .5
    assert report["explicit"]["tp"] == 2 and report["explicit"]["fp"] == 1 and report["explicit"]["fn"] == 2
    assert report["invalid_output_rate"] == .75
    assert report["per_view"]["question"]["n"] == 2
    assert report["usage"]["completion_tokens"] == 16
    assert evaluate([], [], [], catalog)["explicit"]["n"] == 0
    for changed in (predictions[:-1], [*predictions, predictions[0]]):
        with pytest.raises(ValueError, match="exactly one"):
            evaluate(examples, changed, examples, catalog)


class FakeAdapter:
    def __init__(self, fail=False):
        self.fail, self.closed, self.steps = fail, False, 0

    def start(self, config):
        return {"training_session_id": "ts-fake"}

    def use_sampler(self, checkpoint=None):
        self.sampling_checkpoint = checkpoint

    def predict(self, example, config):
        return {"text": json.dumps(GOLD), "latency_seconds": .1, "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "raw": {"choices": []}}

    def train_batch(self, batch, config):
        if self.fail:
            raise RuntimeError("secret in SDK exception")
        self.steps += 1
        return {"loss:sum": 4.0}

    def checkpoint(self, name, sampler=False):
        return name + "-checkpoint"

    def retain_checkpoint(self, path, name):
        return {"name": name, "public": False}

    def close(self):
        self.closed = True


class EpochAdapter(FakeAdapter):
    def predict(self, example, config):
        text = json.dumps(GOLD) if self.sampling_checkpoint == "epoch1-eval-checkpoint" else "garbage"
        return {"text": text, "latency_seconds": .1,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "raw": {}}

    def retain_checkpoint(self, path, name):
        self.retained_path = path
        return super().retain_checkpoint(path, name)


class ResumeAdapter(FakeAdapter):
    def start(self, config, *, resume_state=None):
        self.resume_state = resume_state
        return {"training_session_id": "ts-fake", "run_id": "run-fake"}

    def checkpoint(self, name, sampler=False):
        return f"account/run-fake/{name}{'-sampler' if sampler else ''}"

    def train_batch(self, batch, config):
        self.batches = getattr(self, "batches", []) + [[row.fingerprint() for row in batch]]
        return super().train_batch(batch, config)

    def predict(self, example, config):
        good = self.resume_state is None or "resume-base" in str(self.sampling_checkpoint)
        return {"text": json.dumps(GOLD) if good else "garbage", "latency_seconds": .1,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "raw": {}}


def resume_prepared_pair(prepared):
    config = recipe()
    config["run_id"] = "resume-source"
    config["selection"]["validation"] = 2
    config["training"]["epochs"] = 1
    config["evaluation"]["every_epoch"] = True
    source_run = prepared.run.parent / "resume-source"
    source_manifest = learning_data.prepare_learning(config, ROOT, source_run, prepared.processor, "a" * 40,
        prepared.fetch, {"read_only": True}, renderer_factory)
    continuation = deepcopy(config)
    continuation["run_id"] = "resume-target"
    continuation["selection"]["final_validation"] = 2
    target_run = prepared.run.parent / "resume-target"
    target_manifest = learning_data.prepare_learning(continuation, ROOT, target_run, prepared.processor,
        "a" * 40, prepared.fetch, {"read_only": True}, renderer_factory)
    source = NS(run=source_run, manifest=source_manifest, path=source_run / "manifest.json")
    target = NS(run=target_run, manifest=target_manifest, path=target_run / "manifest.json")
    return source, target


def epoch_prepared(prepared, *, patience=1):
    config = recipe()
    config["run_id"] = "epoch-test"
    config["training"].update(epochs=3, early_stop_patience=patience)
    config["evaluation"]["every_epoch"] = True
    run = prepared.run.parent / "epoch-run"
    manifest = learning_data.prepare_learning(config, ROOT, run, prepared.processor, "a" * 40,
        prepared.fetch, {"read_only": True}, renderer_factory)
    return NS(run=run, manifest=manifest, path=run / "manifest.json")


def execute(prepared, adapter, **kwargs):
    return learning_run.execute_learning(prepared.path, prepared.manifest["run_id"], "a" * 40,
        lambda *args: adapter, lambda model: {"trainingContextLength": 100000},
        renderer_factory=renderer_factory, progress=lambda message: None, **kwargs)


def test_complete_run_compares_identical_cohorts_and_prevents_paid_replay(prepared):
    adapter = FakeAdapter()
    result = execute(prepared, adapter)
    assert result["status"] == "completed" and result["completed_steps"] == 4
    assert adapter.closed and adapter.steps == 4
    comparison = json.loads((prepared.run / "comparison.json").read_text())
    assert comparison["base"]["n"] == comparison["tuned"]["n"] == 4
    assert comparison["delta"]["exact_set_match"] == 0
    with pytest.raises(FileExistsError):
        execute(prepared, adapter)
    assert adapter.steps == 4


def test_compact_run_generates_constrained_predictions_for_every_evaluation(prepared):
    config = recipe()
    config["run_id"] = "compact-evaluation"
    config["selection"].update(validation=2, final_validation=2)
    config["evaluation"].update(every_epoch=True, output_mode="json_schema")
    run = prepared.run.parent / "compact-evaluation"
    manifest = learning_data.prepare_learning(config, ROOT, run, prepared.processor, "a" * 40,
        prepared.fetch, {"read_only": True}, renderer_factory)
    cohort = NS(run=run, manifest=manifest, path=run / "manifest.json")
    adapter = FakeAdapter()
    predict = adapter.predict
    adapter.predict = Mock(side_effect=predict)
    result = execute(cohort, adapter)
    assert result["completed_steps"] == 4
    assert adapter.predict.call_count == 3 * (2 + 2) + 2
    for call in adapter.predict.call_args_list:
        schema = call.args[1]["output_schema"]
        assert "Addition" in schema["properties"]["areas"]["items"]["enum"]
        assert "Addition" not in schema["properties"]["scopes"]["items"]["enum"]
    for name in ("base", "epoch1", "epoch2", "final"):
        assert (run / f"{name}-predictions.json").exists()
    comparison = json.loads((run / "comparison.json").read_text())
    assert comparison["final_validation"]["exact_set_match"] == 1


def test_epoch_selection_and_patience_stop_use_best_validation_checkpoint(prepared):
    cohort = epoch_prepared(prepared)
    adapter = EpochAdapter()
    result = execute(cohort, adapter)
    assert result["completed_steps"] == 4
    assert adapter.retained_path == "epoch1-eval-checkpoint"
    assert any(event["stage"] == "stopped_after_no_validation_gain" for event in result["events"])
    comparison = json.loads((cohort.run / "comparison.json").read_text())
    assert comparison["selected_epoch"] == 1 and comparison["completed_epochs"] == 2
    assert comparison["all_sampling_usage"] == {"prompt_tokens": 180, "completion_tokens": 90}
    assert comparison["base"]["exact_set_match"] == 0
    assert comparison["tuned"]["exact_set_match"] == 1
    assert json.loads((cohort.run / "tuned-predictions.json").read_text()) == json.loads(
        (cohort.run / "epoch1-predictions.json").read_text())
    assert (cohort.run / "epoch2-predictions.json").exists()


def test_cooperative_stop_is_local_and_checked_at_checkpoint(prepared, monkeypatch):
    cohort = epoch_prepared(prepared, patience=3)
    adapter = EpochAdapter()
    with pytest.raises(ValueError, match="active"):
        learning_run.request_learning_stop(cohort.path, cohort.manifest["run_id"])
    with pytest.raises(ValueError, match="confirmation"):
        learning_run.request_learning_stop(cohort.path, "wrong")
    events = []
    def progress(message):
        event = json.loads(message)
        events.append(event)
        if event["stage"] == "training_state_saved":
            request = learning_run.request_learning_stop(cohort.path, cohort.manifest["run_id"])
            assert request["action"] == "stop_at_next_checkpoint"
    result = learning_run.execute_learning(cohort.path, cohort.manifest["run_id"], "a" * 40,
        lambda *args: adapter, lambda model: {"trainingContextLength": 100000},
        renderer_factory=renderer_factory, progress=progress)
    assert result["completed_steps"] == 2
    assert any(event["stage"] == "stop_requested_at_checkpoint" for event in events)
    with pytest.raises(ValueError, match="active"):
        learning_run.request_learning_stop(cohort.path, cohort.manifest["run_id"])
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    args = NS(action="request-stop", manifest=str(cohort.path), confirm_run_id=cohort.manifest["run_id"])
    with pytest.raises(ValueError, match="active"):
        learning_cli.run_learning_command(args, None)


def test_resume_restores_optimizer_and_selects_saved_baseline(prepared, monkeypatch):
    source, target = resume_prepared_pair(prepared)
    source_adapter = ResumeAdapter()
    execute(source, source_adapter)
    adapter = ResumeAdapter()
    record = execute(target, adapter, resume_execution=source.run / "execution.json", resume_epoch=1)
    assert adapter.resume_state == "account/run-fake/epoch1"
    assert record["resume"]["source_run_id"] == "resume-source"
    assert any(event["stage"] == "training_state_restored" for event in record["events"])
    assert [event["epoch"] for event in record["events"] if event["stage"] == "optimizer_step"] == [2, 2]
    assert adapter.steps == 2 and adapter.closed
    _, rows = learning_data.verify_prepared(target.path, "a" * 40)
    expected = [row["render_sha256"] for row in rows if row["split"] == "train"]
    random.Random(recipe()["training"]["seed"] + 1).shuffle(expected)
    assert [item for batch in adapter.batches for item in batch] == expected
    comparison = json.loads((target.run / "comparison.json").read_text())
    assert comparison["selected_epoch"] == 1 and comparison["completed_epochs"] == 1
    assert comparison["cumulative_epochs"] == 2 and comparison["baseline_kind"] == "resumed_training_state"
    assert comparison["base"]["exact_set_match"] == comparison["tuned"]["exact_set_match"] == 1
    assert comparison["final_validation"]["exact_set_match"] == 1
    assert json.loads((target.run / "tuned-predictions.json").read_text()) == json.loads(
        (target.run / "base-predictions.json").read_text())
    with pytest.raises(FileExistsError):
        execute(target, adapter, resume_execution=source.run / "execution.json", resume_epoch=1)
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    args = cli.build_parser().parse_args(["training-api", "resume-check", "--manifest", str(target.path),
        "--source-execution", str(source.run / "execution.json"), "--source-epoch", "1",
        "--confirm-run-id", "resume-target"])
    assert learning_cli.run_learning_command(args, "a" * 40)[1]["status"] == "resume_ready"


def test_resume_rejects_mismatched_cohort_policy_and_source_state(prepared):
    source, target = resume_prepared_pair(prepared)
    execute(source, ResumeAdapter())
    _, rows = learning_data.verify_prepared(target.path, "a" * 40)
    source_path = source.run / "execution.json"
    assert verify_resume_source(target.manifest, rows, source_path, 1)["training_state_path"] == "account/run-fake/epoch1"
    for field, value in (("model", {"id": "other"}), ("training", {"learning_rate": 2e-5}),
                         ("selection", {"seed": 99}), ("evaluation", {"output_mode": "json_schema"})):
        changed = deepcopy(target.manifest)
        changed["recipe"][field].update(value)
        with pytest.raises(ValueError, match="resume"):
            verify_resume_source(changed, rows, source_path, 1)
    bad_rows = deepcopy(rows)
    bad_rows[0]["image_sha256"] = "different"
    with pytest.raises(ValueError, match="examples differ"):
        verify_resume_source(target.manifest, bad_rows, source_path, 1)
    with pytest.raises(ValueError, match="training-state checkpoint"):
        verify_resume_source(target.manifest, rows, source_path, 2)
    with pytest.raises(ValueError, match="positive integer"):
        verify_resume_source(target.manifest, rows, source_path, 0)
    same_id = deepcopy(target.manifest)
    same_id["run_id"] = source.manifest["run_id"]
    with pytest.raises(ValueError, match="new run_id"):
        verify_resume_source(same_id, rows, source_path, 1)
    no_epoch_scoring = deepcopy(target.manifest)
    no_epoch_scoring["recipe"]["evaluation"]["every_epoch"] = False
    with pytest.raises(ValueError, match="every-epoch"):
        verify_resume_source(no_epoch_scoring, rows, source_path, 1)
    wrong_runtime = deepcopy(target.manifest)
    wrong_runtime["language_target_audit"] = {"changed": True}
    with pytest.raises(ValueError, match="runtime"):
        verify_resume_source(wrong_runtime, rows, source_path, 1)
    wrong_processor = deepcopy(target.manifest)
    wrong_processor["files_sha256"]["processor/config.json"] = "different"
    with pytest.raises(ValueError, match="processor"):
        verify_resume_source(wrong_processor, rows, source_path, 1)
    wrong_prompt = deepcopy(target.manifest)
    wrong_prompt["files_sha256"]["prompt.json"] = "different"
    with pytest.raises(ValueError, match="prompt"):
        verify_resume_source(wrong_prompt, rows, source_path, 1)
    conflict_rows = deepcopy(rows)
    train_rows = [row for row in conflict_rows if row["split"] == "train"]
    train_rows[1]["image_sha256"] = train_rows[0]["image_sha256"]
    train_rows[1]["target"] = '{"areas":["Subtraction"],"scopes":[],"abilities":[]}'
    with pytest.raises(ValueError, match="conflicting identical"):
        verify_resume_source(target.manifest, conflict_rows, source_path, 1)
    execution = json.loads(source_path.read_text())
    execution["manifest_sha256"] = "different"
    learning_data.write_json(source_path, execution)
    with pytest.raises(ValueError, match="no longer matches"):
        verify_resume_source(target.manifest, rows, source_path, 1)
    execution["manifest_sha256"] = learning_data.file_sha256(source.path)
    execution["events"] = [{**event, "path": "unqualified"} if event.get("stage") == "training_state_saved" else event
                           for event in execution["events"]]
    learning_data.write_json(source_path, execution)
    with pytest.raises(ValueError, match="fully qualified"):
        verify_resume_source(target.manifest, rows, source_path, 1)
    execution = json.loads(source_path.read_text())
    execution["status"] = "failed"
    learning_data.write_json(source_path, execution)
    with pytest.raises(ValueError, match="completed"):
        verify_resume_source(target.manifest, rows, source_path, 1)


def test_resume_requires_both_source_arguments_before_provider_start(prepared):
    factory = Mock()
    with pytest.raises(ValueError, match="together"):
        learning_run.execute_learning(prepared.path, prepared.manifest["run_id"], "a" * 40,
            factory, Mock(), resume_epoch=1, renderer_factory=renderer_factory)
    factory.assert_not_called()


def test_failed_run_closes_and_never_logs_exception_payload(prepared):
    adapter = FakeAdapter(fail=True)
    with pytest.raises(RuntimeError):
        execute(prepared, adapter)
    record = (prepared.run / "execution.json").read_text()
    assert "secret" not in record and "RuntimeError" in record
    assert adapter.closed and json.loads(record)["status"] == "failed"


@pytest.mark.parametrize("mode", ["confirmation", "ontology", "context", "render"])
def test_launch_guards_fail_before_provider_start(prepared, mode):
    if mode == "ontology":
        prepared.manifest["ontology_snapshot_sha256"] = "wrong"
        learning_data.write_json(prepared.path, prepared.manifest)
    factory = Mock()
    renderer = renderer_factory()
    if mode == "render":
        renderer.system = "wrong"
    with pytest.raises(ValueError):
        learning_run.execute_learning(prepared.path, "bad" if mode == "confirmation" else prepared.manifest["run_id"], "a" * 40,
            factory, lambda model: {"trainingContextLength": 1 if mode == "context" else 100000},
            renderer_factory=lambda *args: renderer if mode == "render" else renderer_factory(*args))
    factory.assert_not_called()


def test_live_capability_check_requires_explicit_vision_v2_sft(monkeypatch):
    data = {"name": "model", "state": "READY", "supportsImageInput": True, "supportsLora": True, "useTrainingV2": True, "supervisedLoraTunable": True}
    client = Mock()
    client.get.return_value.json.return_value = data
    assert inspect_training_model("model", "test-key", client=client)["state"] == "READY"
    manager = Mock(__enter__=Mock(return_value=client), __exit__=Mock(return_value=False))
    monkeypatch.setattr("edugraph_classify.providers.fireworks_training_api.httpx.Client", lambda **kw: manager)
    assert inspect_training_model("model", "test-key")["supportsImageInput"]
    for field in ("supportsImageInput", "supportsLora", "useTrainingV2", "supervisedLoraTunable", "state", "name"):
        client.get.return_value.json.return_value = {**data, field: None}
        with pytest.raises(ValueError, match="live catalog"):
            inspect_training_model("model", "test-key", client=client)


def sdk_adapter():
    future = lambda value: NS(result=lambda **kwargs: value)
    training = NS(run_id="run-fake", forward_backward=Mock(return_value=future(NS(metrics={"loss:sum": 2.0}))),
        optim_step=Mock(return_value=future(NS(metrics={}))), save_state=Mock(return_value=future(NS(path="state"))),
        save_weights_for_sampler=Mock(return_value=future(NS(path="account/run/final-cp"))))
    async def completion(**kwargs):
        assert kwargs["images"][0].startswith("data:image/jpeg;base64,")
        return {"choices": [{"text": json.dumps(GOLD), "finish_reason": "stop"}], "usage": {"prompt_tokens": 20}}, None
    sampler = NS(close=Mock(), deployment_sampler=NS(async_completions_stream=completion))
    service = NS(create_lora_training_client=Mock(return_value=training),
        create_training_client_from_state_with_optimizer=Mock(return_value=training), training_session_id="ts-fake",
        training_session_name="accounts/a/trainingSessions/ts-fake", create_sampling_client=Mock(return_value=sampler), close=Mock(), holder=NS(close=Mock()))
    platform = NS(list_training_session_checkpoints=Mock(return_value=[{"name": "accounts/a/trainingSessions/ts-fake/checkpoints/run-final-cp", "promotable": True}]),
                  promote_session_checkpoint=Mock(return_value={"name": "model", "public": False}))
    platform_context = Mock(__enter__=Mock(return_value=platform), __exit__=Mock(return_value=False))
    types = NS(**{k: (lambda **kwargs: NS(**kwargs)) for k in ("Datum", "ModelInput", "EncodedTextChunk", "ImageChunk", "TensorData", "AdamParams")})
    adapter = FireworksTrainingAPI("test-key", "model", Tokenizer(), service_factory=lambda **k: service, platform_factory=lambda **k: platform_context, types=types)
    return adapter, service, training, platform, types


def test_sdk_adapter_transmits_multimodal_chunks_and_cleans_all_clients():
    adapter, service, training, platform, types = sdk_adapter()
    example = renderer_factory().render(png(), json.dumps(GOLD), max_tokens=10000, max_output=20)
    datum = to_datum(example, types)
    assert datum.model_input.chunks[1].expected_tokens == 7
    assert sum(datum.loss_fn_inputs["weights"].data) == len(example.completion)
    assert adapter.start(recipe()["training"])["run_id"] == "run-fake"
    assert service.create_lora_training_client.call_args.kwargs["train_attn"] is True
    assert adapter.train_batch([example], recipe()["training"])["loss:sum"] == 2
    assert training.optim_step.call_count == 1
    assert adapter.checkpoint("epoch1") == "state"
    checkpoint = adapter.checkpoint("final", sampler=True)
    adapter.use_sampler()
    assert adapter.predict(example, recipe()["evaluation"])["finish_reason"] == "stop"
    adapter.use_sampler(checkpoint)
    assert adapter.retain_checkpoint(checkpoint, "output")["public"] is False
    platform.list_training_session_checkpoints.return_value = []
    with pytest.raises(ValueError, match="retention"):
        adapter.retain_checkpoint(checkpoint, "output")
    training.forward_backward.return_value = NS(result=lambda **k: NS(metrics={"loss:sum": float("nan")}))
    with pytest.raises(ValueError, match="non-finite"):
        adapter.train_batch([example], recipe()["training"])
    assert training.optim_step.call_count == 1
    adapter.close()
    service.close.assert_called_once()
    service.holder.close.assert_called_once()
    adapter, *_ = sdk_adapter()
    adapter.close()


def test_sdk_adapter_passes_closed_schema_and_checks_retained_target_modules():
    adapter, service, _, platform, _ = sdk_adapter()
    config = {**recipe()["training"], "expected_target_modules": ["down_proj"],
              "train_attn": True, "train_mlp": True, "train_unembed": False}
    adapter.start(config)
    assert service.create_lora_training_client.call_args.kwargs["train_unembed"] is False
    adapter.use_sampler()
    example = renderer_factory().render(png(), json.dumps(GOLD), max_tokens=10000, max_output=20)
    schema = {"type": "object", "properties": {"areas": {"type": "array"}}}
    received = {}
    async def sample(**kwargs):
        received.update(kwargs)
        return {"choices": [{"text": json.dumps(GOLD), "finish_reason": "stop"}]}, None
    adapter.sampler.deployment_sampler.async_completions_stream = sample
    adapter.predict(example, {**recipe()["evaluation"], "output_schema": schema})
    assert received["response_format"] == {"type": "json_schema", "json_schema": {"name": "edugraph_labels", "schema": schema}}
    platform.promote_session_checkpoint.return_value = {"name": "model", "peftDetails": {"targetModules": ["down_proj"]}}
    assert adapter.retain_checkpoint("account/run/final-cp", "out")["name"] == "model"
    platform.promote_session_checkpoint.return_value["peftDetails"]["targetModules"] = ["qkv"]
    with pytest.raises(ValueError, match="target modules"):
        adapter.retain_checkpoint("account/run/final-cp", "out")
    adapter.close()


def test_sdk_resume_uses_training_state_and_optimizer_instead_of_new_lora():
    adapter, service, _, _, _ = sdk_adapter()
    result = adapter.start(recipe()["training"], resume_state="account/run-source/epoch1")
    service.create_training_client_from_state_with_optimizer.assert_called_once_with("account/run-source/epoch1")
    service.create_lora_training_client.assert_not_called()
    assert result["resumed_from_training_state"] == "account/run-source/epoch1"
    adapter.close()


def test_default_sdk_imports_are_lazy_and_do_not_start_compute(monkeypatch):
    monkeypatch.setitem(sys.modules, "fireworks.training.sdk", NS(FiretitanServiceClient=Mock()))
    monkeypatch.setitem(sys.modules, "fireworks.training.sdk.fireworks_client", NS(FireworksClient=Mock()))
    monkeypatch.setitem(sys.modules, "tinker", NS(types=NS()))
    adapter = FireworksTrainingAPI("fake-key", "model", Tokenizer())
    assert adapter.service is None


def test_cli_prepare_and_launch_wiring_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(learning_cli, "load_local_environment", lambda path: None)
    monkeypatch.setattr(learning_cli, "inspect_training_model", lambda *args: {"ok": True})
    monkeypatch.setitem(sys.modules, "huggingface_hub", NS(snapshot_download=Mock()))
    client = Mock()
    client.get.return_value.content = b"test"
    monkeypatch.setattr(learning_cli.httpx, "Client", lambda **kwargs: Mock(__enter__=Mock(return_value=client), __exit__=Mock(return_value=False)))
    def prepare(*args):
        assert args[5]("https://example.test") == b"test"
        return {"selected_images_validated": 224, "optimizer_steps": 120, "estimated_max_token_cost_usd": 8}
    monkeypatch.setattr(learning_cli, "prepare_learning", prepare)
    args = cli.build_parser().parse_args(["training-api", "prepare", "--config", str(RECIPE)])
    assert learning_cli.run_learning_command(args, "a" * 40)[1]["images"] == 224
    assert logging.root.manager.disable == 0
    def execute(path, confirmation, commit, factory, inspect, **kwargs):
        monkeypatch.setattr(learning_cli, "FireworksTrainingAPI", lambda *args: "adapter")
        assert factory("model", "tokenizer") == "adapter"
        assert inspect("model") == {"ok": True}
        if confirmation == "new":
            assert kwargs["resume_execution"] == Path("runs/old/execution.json")
            assert kwargs["resume_epoch"] == 1
        kwargs["progress"]("progress")
        return {"status": "completed", "completed_steps": 120, "retained_model": {}}
    monkeypatch.setattr(learning_cli, "execute_learning", execute)
    args = cli.build_parser().parse_args(["training-api", "launch", "--manifest", "runs/x/manifest.json", "--confirm-run-id", "x"])
    assert learning_cli.run_learning_command(args, "a" * 40)[1]["completed_steps"] == 120
    args = cli.build_parser().parse_args(["training-api", "resume", "--manifest", "runs/new/manifest.json",
        "--source-execution", "runs/old/execution.json", "--source-epoch", "1", "--confirm-run-id", "new"])
    assert learning_cli.run_learning_command(args, "a" * 40)[1]["completed_steps"] == 120
    monkeypatch.setattr(cli, "_code_commit", lambda root: "a" * 40)
    monkeypatch.setattr(cli, "run_learning_command", lambda *args: (0, {"status": "prepared"}))
    assert cli.main(["training-api", "prepare"]) == 0
    bad = recipe()
    bad["model"]["provider"] = "elsewhere"
    monkeypatch.setattr(learning_cli, "load_recipe", lambda path: bad)
    args.action = "prepare"
    args.config = str(RECIPE)
    with pytest.raises(ValueError, match="only the Fireworks"):
        learning_cli.run_learning_command(args, "a" * 40)
