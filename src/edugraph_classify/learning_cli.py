"""CLI wiring: public downloads, explicit cost confirmation, and SDK isolation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx

from .configuration import load_local_environment
from .learning_data import load_recipe, prepare_learning, verify_prepared
from .learning_run import execute_learning, request_learning_stop
from .learning_resume import verify_resume_source
from .providers.fireworks_training_api import FireworksTrainingAPI, inspect_training_model


def add_learning_parser(commands) -> None:
    parser = commands.add_parser("training-api", help="prepare or run a bounded training API experiment")
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare", help="read-only eligibility check and local data preparation")
    prepare.add_argument("--config", default="experiments/edugraph-20260928-qwen38-27b-learning-v1.json")
    prepare.add_argument("--runs-root", default="runs")
    prepare.add_argument("--env-file", default=".env")
    launch = actions.add_parser("launch", help="run one explicitly confirmed paid experiment")
    launch.add_argument("--manifest", required=True)
    launch.add_argument("--confirm-run-id", required=True)
    launch.add_argument("--env-file", default=".env")
    for action, help_text in (("resume-check", "validate a continuation locally without provider calls"),
                              ("resume", "run one explicitly confirmed paid continuation")):
        continuation = actions.add_parser(action, help=help_text)
        continuation.add_argument("--manifest", required=True, help="new prepared run manifest")
        continuation.add_argument("--source-execution", required=True, help="completed source run execution.json")
        continuation.add_argument("--source-epoch", required=True, type=int, help="saved epoch to resume from")
        continuation.add_argument("--confirm-run-id", required=True, help="new prepared run ID")
        if action == "resume":
            continuation.add_argument("--env-file", default=".env")
    stop = actions.add_parser("request-stop", help="finish the active run at its next durable epoch checkpoint")
    stop.add_argument("--manifest", required=True)
    stop.add_argument("--confirm-run-id", required=True)


def run_learning_command(args, code_commit: str | None) -> tuple[int, dict]:
    if args.action == "request-stop":
        request = request_learning_stop(Path(args.manifest), args.confirm_run_id)
        return 0, {"status": "stop_requested", **request}
    if code_commit is None:
        raise ValueError("preparation and launch require a pinned clean commit")
    if args.action == "resume-check":
        manifest, examples = verify_prepared(Path(args.manifest), code_commit)
        if args.confirm_run_id != manifest["run_id"]:
            raise ValueError("confirmation must exactly match the new prepared run_id")
        source = verify_resume_source(manifest, examples, Path(args.source_execution), args.source_epoch)
        return 0, {"status": "resume_ready", "new_run_id": manifest["run_id"], **source,
                   "additional_epochs": manifest["recipe"]["training"]["epochs"],
                   "estimated_max_token_cost_usd": manifest["estimated_max_token_cost_usd"]}
    load_local_environment(args.env_file)
    key = os.environ["FIREWORKS_API_KEY"]
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        if args.action == "prepare":
            from huggingface_hub import snapshot_download

            config = load_recipe(Path(args.config))
            model = config["model"]
            if model["provider"] != "fireworks" or model["execution_mode"] != "serverless":
                raise ValueError("this command currently supplies only the Fireworks serverless adapter")
            capabilities = inspect_training_model(model["id"], key)
            processor = Path("temp") / f"processor-{model['hf_revision']}"
            snapshot_download(model["hf_repository"], revision=model["hf_revision"], token=False,
                              local_dir=processor, allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                              "vocab.json", "merges.txt", "*processor_config.json", "chat_template*"])
            run = Path(args.runs_root) / config["run_id"]
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                def fetch(url):
                    response = client.get(url)
                    response.raise_for_status()
                    return response.content
                manifest = prepare_learning(config, Path.cwd(), run, processor, code_commit, fetch, capabilities)
            return 0, {"status": "prepared", "manifest": str(run / "manifest.json"),
                       "images": manifest["selected_images_validated"], "optimizer_steps": manifest["optimizer_steps"],
                       "estimated_max_token_cost_usd": manifest["estimated_max_token_cost_usd"]}
        record = execute_learning(Path(args.manifest), args.confirm_run_id, code_commit,
            lambda model, tokenizer: FireworksTrainingAPI(key, model, tokenizer),
            lambda model: inspect_training_model(model, key),
            resume_execution=Path(args.source_execution) if args.action == "resume" else None,
            resume_epoch=args.source_epoch if args.action == "resume" else None,
            progress=lambda message: print(message, flush=True))
        return 0, {"status": record["status"], "completed_steps": record["completed_steps"],
                   "retained_model": record["retained_model"], "resume": record.get("resume")}
    finally:
        logging.disable(previous_logging)
