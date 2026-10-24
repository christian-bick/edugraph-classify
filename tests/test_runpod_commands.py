from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock
import json

import pytest

from edugraph_classify import cli, runpod_cli
from edugraph_classify.runpod_config import load_runpod_recipe
from edugraph_classify.runpod_recipe import new_release_recipe, write_new_recipe

ROOT = Path(__file__).resolve().parents[1]


def new_recipe(**overrides):
    values = dict(run_id="next-release", dataset_release="v0.32.0-01",
                  dataset_revision="b" * 40, ontology_version="0.32.0",
                  estimated_hours=10, max_hours=12, cost_limit_usd=8)
    return new_release_recipe(load_runpod_recipe(ROOT / runpod_cli.DEFAULT_RECIPE),
                              **(values | overrides))


def test_new_release_preserves_validated_policy_and_uses_whole_release(tmp_path):
    template = load_runpod_recipe(ROOT / runpod_cli.DEFAULT_RECIPE)
    original = deepcopy(template)
    result = new_recipe(context_tokens=8192)
    assert template == original
    assert result["model"] == original["model"]
    assert result["training"] == original["training"] | {"max_context_tokens": 8192}
    assert result["dataset"]["source_ontology_version"] == result["ontology_version"] == "0.32.0"
    assert result["selection"]["train"] == "all"
    assert result["selection"]["final_validation"] == "remaining"
    path = tmp_path / "experiments/new.json"
    write_new_recipe(path, result)
    assert load_runpod_recipe(path) == result
    with pytest.raises(FileExistsError):
        write_new_recipe(path, result | {"run_id": "replacement"})
    assert json.loads(path.read_text()) == result


@pytest.mark.parametrize("override", [
    {"run_id": "bad/id"}, {"dataset_revision": "main"}, {"dataset_release": ""},
    {"ontology_version": ""}, {"context_tokens": 0}, {"context_tokens": True},
    {"estimated_hours": 13}, {"cost_limit_usd": 0.01}, {"estimated_hours": float("nan")},
])
def test_new_release_rejects_invalid_identity_or_budget(override):
    with pytest.raises(ValueError):
        new_recipe(**override)


def test_new_release_rejects_reusing_template_run_identity():
    with pytest.raises(ValueError, match="new run ID"):
        new_recipe(run_id=load_runpod_recipe(ROOT / runpod_cli.DEFAULT_RECIPE)["run_id"])


def no_git(monkeypatch):
    monkeypatch.setattr(cli, "_code_commit", Mock(side_effect=AssertionError("must not need a clean Git checkout")))


def test_new_run_cli_does_not_require_commit_or_provider(tmp_path, monkeypatch, capsys):
    no_git(monkeypatch)
    output = tmp_path / "new.json"
    assert cli.main(["runpod", "new-run", "--run-id", "next-run", "--dataset-release", "v0.32.0-01",
                     "--dataset-revision", "a" * 40, "--ontology-version", "0.32.0",
                     "--estimated-hours", "10", "--max-hours", "12", "--cost-limit-usd", "8",
                     "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "recipe_created"
    assert load_runpod_recipe(output)["run_id"] == "next-run"


def test_retrieve_and_verify_cli_are_read_only_and_save_report(tmp_path, monkeypatch, capsys):
    no_git(monkeypatch)
    monkeypatch.setattr(runpod_cli, "load_local_environment", lambda _: None)
    store = Mock()
    monkeypatch.setattr(runpod_cli.GcsBundles, "from_environment", lambda: store)
    retrieve = Mock(return_value={"run_id": "one"})
    monkeypatch.setattr(runpod_cli, "retrieve_result", retrieve)
    manifest = tmp_path / "manifest.json"
    assert cli.main(["runpod", "retrieve", "--manifest", str(manifest), "--output", str(tmp_path)]) == 0
    retrieve.assert_called_once_with(manifest, tmp_path, store)
    assert json.loads(capsys.readouterr().out)["status"] == "retrieved"
    completion = {"run_id": "one"}
    (tmp_path / "completion.json").write_text(json.dumps(completion))
    (tmp_path / "retrieval.json").write_text(json.dumps({"model_dir": str(tmp_path / "model"),
                                                       "checkpoint_dir": str(tmp_path / "checkpoint")}))
    verify = Mock(return_value={"run_id": "one", "all_metrics_reproduced": True})
    monkeypatch.setattr(runpod_cli, "verify_retrieved", verify)
    assert cli.main(["runpod", "verify-result", "--manifest", str(manifest), "--result-dir", str(tmp_path)]) == 0
    verify.assert_called_once_with(manifest, tmp_path)
    assert json.loads((tmp_path / "verified-result.json").read_text())["all_metrics_reproduced"]
    assert json.loads(capsys.readouterr().out)["status"] == "verified"


def test_status_can_query_publication_without_runpod_credentials(tmp_path, monkeypatch, capsys):
    no_git(monkeypatch)
    monkeypatch.setattr(runpod_cli, "load_local_environment", lambda _: None)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    store, client = Mock(), Mock()
    monkeypatch.setattr(runpod_cli.GcsBundles, "from_environment", lambda: store)
    monkeypatch.setattr(runpod_cli, "RunpodClient", lambda _: client)
    inspect = Mock(return_value={"status": "completed", "artifacts_verified": False})
    monkeypatch.setattr(runpod_cli, "inspect_run", inspect)
    manifest, launch = tmp_path / "manifest.json", tmp_path / "launch.json"
    manifest.write_text('{"run_id": "one"}')
    launch.write_text('{"run_id": "one", "pod": {"id": "pod1"}}')
    arguments = ["runpod", "status", "--manifest", str(manifest)]
    assert cli.main(arguments) == 0
    inspect.assert_called_with({"run_id": "one"}, store, None, None)
    assert not json.loads(capsys.readouterr().out)["artifacts_verified"]
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-for-test")
    assert cli.main(arguments + ["--launch-record", str(launch)]) == 0
    inspect.assert_called_with({"run_id": "one"}, store, client, json.loads(launch.read_text()))
    assert cli.main(["runpod", "status"]) == 2
    assert "requires --manifest or --launch-record" in capsys.readouterr().err


def test_dry_preparation_is_explicitly_unlaunchable(tmp_path, monkeypatch, capsys):
    no_git(monkeypatch)
    prepare = Mock(return_value={"estimated_run_cost_usd": 5.14})
    monkeypatch.setattr(runpod_cli, "prepare_runpod", prepare)
    assert cli.main(["runpod", "prepare", "--config", runpod_cli.DEFAULT_RECIPE,
                     "--runs-root", str(tmp_path), "--dry-run"]) == 0
    assert prepare.call_args.args[2] == "uncommitted-dry-run"
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "dry_run_prepared" and response["launchable"] is False


def test_commit_guard_rejects_wrong_editable_checkout_and_dirty_source(tmp_path, monkeypatch):
    source = tmp_path / "src/edugraph_classify/cli.py"
    source.parent.mkdir(parents=True)
    source.write_text("different checkout")
    with pytest.raises(ValueError, match="another checkout"):
        cli._code_commit(tmp_path)
    monkeypatch.setattr(cli, "__file__", str(source))
    git = Mock(return_value=Mock(stdout=" M source.py"))
    monkeypatch.setattr(cli.subprocess, "run", git)
    with pytest.raises(ValueError, match="clean committed"):
        cli._code_commit(tmp_path)
    git.side_effect = [Mock(stdout=""), Mock(stdout="a" * 40 + "\n")]
    assert cli._code_commit(tmp_path) == "a" * 40


def test_decoder_can_represent_more_than_twenty_labels_in_a_dimension():
    from importlib.metadata import version
    from lmformatenforcer import JsonSchemaParser
    from edugraph_classify.ontology import OntologyCatalog
    from edugraph_classify.output_contract import closed_vocabulary_schema

    catalog = OntologyCatalog.load(version("edugraph-py"))
    schema = closed_vocabulary_schema(json.loads((ROOT / "schemas/dimension-labels-v1.json").read_text()), catalog)
    labels = schema["properties"]["areas"]["items"]["enum"][:21]
    assert len(labels) == 21
    parser = JsonSchemaParser(schema)
    text = json.dumps({"areas": labels, "scopes": [], "abilities": []}, separators=(",", ":"))
    for char in text:
        assert char in parser.get_allowed_characters()
        parser = parser.add_character(char)
    assert parser.can_end()
