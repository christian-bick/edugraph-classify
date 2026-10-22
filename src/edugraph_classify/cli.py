"""Runpod training and optional inference benchmark command entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .providers.runpod import RunpodError
from .hf_gguf_release import add_hf_gguf_release_parser, run_hf_gguf_release_command
from .inference_benchmark_cli import add_inference_benchmark_parser, run_inference_benchmark_command
from .runpod_cli import add_runpod_parser, run_runpod_command


def _code_commit(repo_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("run preparation requires a clean committed worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edugraph-classify")
    commands = parser.add_subparsers(dest="command", required=True)
    add_runpod_parser(commands)
    add_inference_benchmark_parser(commands)
    add_hf_gguf_release_parser(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "hf-gguf-release":
            exit_code, payload = run_hf_gguf_release_command(args)
            print(json.dumps(payload, indent=2))
            return exit_code
        commit = None if args.action in ("check-config", "compare", "status", "stop") else _code_commit(Path.cwd())
        if args.command == "inference-benchmark":
            exit_code, payload = run_inference_benchmark_command(args, commit)
        else:
            exit_code, payload = run_runpod_command(args, commit)
    except (ValueError, RunpodError) as error:
        print(json.dumps({"status": "error", "message": str(error)}, indent=2), file=sys.stderr)
        return 2
    except Exception as error:  # Provider errors must not expose request configuration.
        print(json.dumps({"status": "error", "message": f"command failed with {type(error).__name__}"}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))
    return exit_code
