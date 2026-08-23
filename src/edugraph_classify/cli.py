"""Explicit read-only preflight entry points."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .configuration import load_local_environment
from .preflight import EligibilityStatus, PreflightError
from .providers.fireworks import FireworksTrainingProvider


def run_fireworks_preflight(
    hypothesis: str,
    env_file: str,
    *,
    client: Any | None = None,
) -> tuple[int, dict[str, object]]:
    load_local_environment(env_file)
    if client is None:
        from fireworks import Fireworks

        client = Fireworks()
    result = FireworksTrainingProvider(client).preflight_model(hypothesis)
    exit_code = 0 if result.status is EligibilityStatus.ELIGIBLE else 2
    return exit_code, result.to_mapping()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edugraph-classify")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="run read-only provider capability checks")
    targets = preflight.add_subparsers(dest="target", required=True)

    fireworks = targets.add_parser("fireworks", help="inspect a Fireworks model hypothesis")
    fireworks.add_argument("--hypothesis", default="Qwen3.5-9B")
    fireworks.add_argument("--env-file", default=".env")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        exit_code, payload = run_fireworks_preflight(args.hypothesis, args.env_file)
    except PreflightError as error:
        print(json.dumps({"status": "error", "message": str(error)}, indent=2), file=sys.stderr)
        return 2
    except Exception as error:  # Provider SDK errors must never expose request configuration.
        payload = {
            "status": "error",
            "message": f"preflight failed with {type(error).__name__}",
        }
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return exit_code
