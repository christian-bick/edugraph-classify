from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from edugraph_classify import cli
from edugraph_classify.providers.fireworks import ModelResolutionError


def test_fireworks_preflight_loads_env_and_returns_fail_closed_status(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FIREWORKS_API_KEY=test-only\n", encoding="utf-8")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    model = SimpleNamespace(
        name="accounts/fireworks/models/qwen3p5-9b",
        display_name="Qwen3.5 9B",
        supports_image_input=True,
        tunable=None,
        base_model_details=SimpleNamespace(tunable=False),
        supports_lora=True,
        supports_serverless=False,
        context_length=262_144,
        training_context_length=131_072,
        model_extra={"supervisedLoraTunable": True, "useTrainingV2": True},
    )

    class Models:
        def list(self, **kwargs: object) -> list[SimpleNamespace]:
            return [model]

        def get(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return model

    exit_code, payload = cli.run_fireworks_preflight(
        "Qwen3.5-9B", str(env_file), client=SimpleNamespace(models=Models())
    )

    assert exit_code == 2
    assert payload["status"] == "ineligible"
    assert payload["training_execution_mode"] == "provider_dedicated"


def test_parser_defaults_to_accepted_provider_hypothesis() -> None:
    args = cli.build_parser().parse_args(["preflight", "fireworks"])
    assert args.hypothesis == "Qwen3.5-9B"
    assert args.env_file == ".env"


def test_main_prints_payload_and_preserves_preflight_exit_code(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "run_fireworks_preflight",
        lambda hypothesis, env_file: (2, {"status": "ineligible", "hypothesis": hypothesis}),
    )

    assert cli.main(["preflight", "fireworks"]) == 2
    assert '"status": "ineligible"' in capsys.readouterr().out


def test_main_reports_safe_known_and_unexpected_errors(monkeypatch, capsys) -> None:
    def known(*args: object) -> object:
        raise ModelResolutionError("missing model")

    monkeypatch.setattr(cli, "run_fireworks_preflight", known)
    assert cli.main(["preflight", "fireworks"]) == 2
    assert "missing model" in capsys.readouterr().err

    def unexpected(*args: object) -> object:
        raise RuntimeError("secret-value-must-not-be-printed")

    monkeypatch.setattr(cli, "run_fireworks_preflight", unexpected)
    assert cli.main(["preflight", "fireworks"]) == 1
    error = capsys.readouterr().err
    assert "RuntimeError" in error
    assert "secret-value" not in error


def test_main_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
