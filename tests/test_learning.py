from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from jinja2 import Environment
from PIL import Image

from edugraph_classify import learning_data, runpod_config, self_hosted_run
from edugraph_classify.dataset import PromptTemplate
from edugraph_classify.evaluation import evaluate, parse_prediction
from edugraph_classify.ontology import OntologyCatalog
from edugraph_classify.output_contract import closed_vocabulary_schema
from edugraph_classify.qwen_training_policy import assess_language_targets
from edugraph_classify.qwen_rendering import IMAGE_MARKER, QwenVisionRenderer, fixed_system_template

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "experiments/edugraph-20261002-qwen38-27b-runpod-full-a40-v1.json"
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
    value = runpod_config.load_runpod_recipe(RECIPE)
    value["selection"] = {"train": 4, "validation": 4, "final_validation": 0,
                          "train_diagnostic": 2, "seed": 42}
    value["training"].update(epochs=2, batch_size=2, max_context_tokens=50000)
    return value


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(learning_data, "audit_qwen_language_targets", lambda *args: {"verified": True})
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
    manifest = learning_data.prepare_learning(recipe(), ROOT, run, processor, "a" * 40, fetch,
                                              {"read_only": True}, renderer_factory,
                                              runtime_packages=("pillow",))
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
        runpod_config.load_runpod_recipe(path)


def test_epoch_evaluation_requires_valid_patience(tmp_path):
    for value in (0, -1, 1.5, True):
        config = recipe()
        config["training"]["early_stop_patience"] = value
        path = tmp_path / "recipe.json"
        learning_data.write_json(path, config)
        with pytest.raises(ValueError, match="early_stop_patience"):
            runpod_config.load_runpod_recipe(path)
    config["evaluation"]["every_epoch"] = True
    config["training"]["early_stop_patience"] = 1
    learning_data.write_json(path, config)
    assert runpod_config.load_runpod_recipe(path)["training"]["early_stop_patience"] == 1


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
        prepared.fetch, {"read_only": True}, renderer_factory,
        runtime_packages=("pillow",))
    assert manifest["cohort_counts"] == {"train": 4, "validation": 2, "final_validation": 2}
    _, rows = learning_data.verify_prepared(run / "manifest.json", "a" * 40)
    selected = {r["id"] for r in rows if r["split"] == "validation"}
    final = {r["id"] for r in rows if r["split"] == "final_validation"}
    assert len(selected | final) == 4 and not selected & final
    expected_generations = ((1 + config["training"]["epochs"])
                            * (config["selection"]["validation"] + config["selection"]["train_diagnostic"])
                            + config["selection"]["final_validation"])
    assert manifest["token_budget"]["sample"] == expected_generations * config["evaluation"]["max_tokens"]


def test_conflicting_identical_train_images_block_runpod_engine(prepared):
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
        fetch, {"read_only": True}, renderer_factory,
        runtime_packages=("pillow",))
    assert manifest["duplicate_image_counts"]["train"] == 1
    assert len(manifest["training_data_conflicts"]) == 1
    assert {row["gold"]["areas"][0] for row in manifest["training_data_conflicts"][0]["examples"]} == {"Addition", "Subtraction"}
    factory = Mock()
    with pytest.raises(ValueError, match="conflicting gold"):
        self_hosted_run.execute_self_hosted(run / "manifest.json", "a" * 40,
            run.parent / "work", factory, Mock(), Mock(), renderer_factory=renderer_factory)
    factory.assert_not_called()
    assert not (run.parent / "work").exists()


def test_preparation_pins_audits_and_detects_tampering(prepared):
    manifest, examples = learning_data.verify_prepared(prepared.path, "a" * 40)
    assert manifest["metadata_rows_validated"] == {"train": 4, "validation": 4}
    assert manifest["selected_images_validated"] == 8
    assert manifest["optimizer_steps"] == 4
    assert len([r for r in examples if r["train_diagnostic"]]) == 2
    assert manifest["estimated_run_cost_usd"] > 0
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


def test_compact_prompt_matches_prepared_embedded_template(prepared):
    prompt = PromptTemplate.load(ROOT / "prompts/direct-label-v3.json")
    assert json.loads((prepared.run / "prompt.json").read_text()) == prompt.to_mapping()
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
        learning_data.prepare_learning(config, ROOT, prepared.run.parent / "bad", prepared.processor,
                                       "a" * 40, fetch, {}, renderer_factory,
                                       runtime_packages=("pillow",))


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
