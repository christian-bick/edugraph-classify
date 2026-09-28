"""Bounded training/evaluation orchestration with a replaceable provider adapter."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .dataset import file_sha256
from .evaluation import evaluate
from .learning_data import verify_prepared, write_json
from .ontology import OntologyCatalog
from .qwen_training_policy import audit_qwen_language_targets
from .qwen_rendering import QwenVisionRenderer


def requested_stop(path: Path, run_id: str) -> dict | None:
    """Read a create-only stop request at a saved checkpoint boundary."""
    if not path.exists():
        return None
    request = json.loads(path.read_text(encoding="utf-8"))
    if request != {"run_id": run_id, "action": "stop_at_next_checkpoint"}:
        raise ValueError("stop request does not match this run")
    return request


def request_learning_stop(manifest_path: Path, confirmation: str) -> dict:
    pinned_commit = json.loads(manifest_path.read_text(encoding="utf-8"))["code_commit"]
    manifest, _ = verify_prepared(manifest_path, pinned_commit)
    if confirmation != manifest["run_id"]:
        raise ValueError("confirmation must match the prepared run_id")
    run = manifest_path.parent
    record_path = run / "execution.json"
    if not record_path.exists() or json.loads(record_path.read_text(encoding="utf-8"))["status"] != "started":
        raise ValueError("a training execution must be active to request a stop")
    request = {"run_id": manifest["run_id"], "action": "stop_at_next_checkpoint"}
    with (run / "stop-request.json").open("x", encoding="utf-8") as output:
        json.dump(request, output)
    return request


def render_prepared(run: Path, config: dict, examples: list[dict], renderer) -> dict:
    rendered = {}
    for row in examples:
        example = renderer.render((run / row["image_path"]).read_bytes(), row["target"],
            max_tokens=config["training"]["max_context_tokens"], max_output=config["evaluation"]["max_tokens"])
        if example.fingerprint() != row["render_sha256"]:
            raise ValueError("rendering differs from the prepared manifest")
        rendered[row["id"]] = example
    return rendered


def execute_learning(manifest_path: Path, confirmation: str, code_commit: str, adapter_factory,
                     inspect_model, *, renderer_factory=QwenVisionRenderer.load, progress=print) -> dict:
    manifest, examples = verify_prepared(manifest_path, code_commit)
    config, run = manifest["recipe"], manifest_path.parent
    if confirmation != manifest["run_id"]:
        raise ValueError("confirmation must exactly match the prepared run_id")
    catalog = OntologyCatalog.load(config["ontology_version"])
    if any(version(name) != pinned for name, pinned in manifest["runtime_versions"].items()):
        raise ValueError("installed runtime differs from preparation; sync the pinned lockfile")
    if catalog.snapshot_sha256 != manifest["ontology_snapshot_sha256"]:
        raise ValueError("ontology snapshot differs from preparation")
    capabilities = inspect_model(config["model"]["id"])
    if not isinstance(capabilities.get("trainingContextLength"), int) or config["training"]["max_context_tokens"] > capabilities["trainingContextLength"]:
        raise ValueError("configured context exceeds live training support")
    prompt = json.loads((run / "prompt.json").read_text(encoding="utf-8"))
    renderer = renderer_factory(run / "processor", prompt)
    if config["training"].get("expected_target_modules") is not None:
        audit = audit_qwen_language_targets(run / "processor", config["training"]["expected_target_modules"])
        if audit != manifest["language_target_audit"]:
            raise ValueError("language-only LoRA target layout changed since preparation")
    rendered = render_prepared(run, config, examples, renderer)
    train = [r for r in examples if r["split"] == "train"]
    validation = [r for r in examples if r["split"] == "validation"]
    diagnostic = [r for r in train if r["train_diagnostic"]]
    evaluations = validation + diagnostic
    sampling_config = dict(config["evaluation"])
    if sampling_config.get("output_mode", "raw") == "json_schema":
        sampling_config["output_schema"] = json.loads((run / "closed_schema.json").read_text(encoding="utf-8"))
    record_path = run / "execution.json"
    # Exclusive creation prevents paid replays, including after ambiguous failures.
    record = {"status": "started", "run_id": manifest["run_id"], "manifest_sha256": file_sha256(manifest_path),
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "capabilities": capabilities,
              "completed_steps": 0, "events": []}
    with record_path.open("x", encoding="utf-8") as output:
        json.dump(record, output)

    def event(stage, **values):
        record["events"].append({"stage": stage, **values})
        write_json(record_path, record)
        progress(json.dumps(record["events"][-1]))

    adapter = adapter_factory(config["model"]["id"], renderer.tokenizer)
    try:
        record["provider"] = adapter.start(config["training"])
        event("session_started", **record["provider"])

        def sample_phase(phase):
            predictions = []
            for index, row in enumerate(evaluations):
                result = {"id": row["id"], **adapter.predict(rendered[row["id"]], sampling_config)}
                predictions.append(result)
                write_json(run / f"{phase}-predictions.json", predictions)
                if (index + 1) % 8 == 0 or index + 1 == len(evaluations):
                    event("evaluation_progress", phase=phase, completed=index + 1, total=len(evaluations))
            metrics = {}
            for name, cohort in (("validation", validation), ("train_diagnostic", diagnostic)):
                ids = {r["id"] for r in cohort}
                metrics[name] = evaluate(cohort, [r for r in predictions if r["id"] in ids], train, catalog)
            write_json(run / f"{phase}-metrics.json", metrics)
            event("evaluation_finished", phase=phase, validation=metrics["validation"]["explicit"],
                  invalid_output_rate=metrics["validation"]["invalid_output_rate"])
            return metrics

        adapter.use_sampler()
        baseline = sample_phase("base")
        batch_size = config["training"]["batch_size"]
        every_epoch = config["evaluation"].get("every_epoch", False)
        best = None
        stale_epochs = 0
        for epoch in range(config["training"]["epochs"]):
            order = list(train)
            random.Random(config["training"]["seed"] + epoch).shuffle(order)
            for offset in range(0, len(order), batch_size):
                batch = [rendered[r["id"]] for r in order[offset:offset + batch_size]]
                metrics = adapter.train_batch(batch, config["training"])
                supervised = sum(len(row.completion) for row in batch)
                record["completed_steps"] += 1
                event("optimizer_step", step=record["completed_steps"], epoch=epoch + 1, metrics=metrics,
                      training_tokens=sum(row.training_tokens for row in batch),
                      supervised_tokens=supervised, mean_nll=metrics.get("loss:sum", 0.0) / supervised)
            state = adapter.checkpoint(f"epoch{epoch + 1}")
            event("training_state_saved", epoch=epoch + 1, path=state)
            if every_epoch:
                epoch_checkpoint = adapter.checkpoint(f"epoch{epoch + 1}-eval", sampler=True)
                adapter.use_sampler(epoch_checkpoint)
                epoch_metrics = sample_phase(f"epoch{epoch + 1}")
                outcome = epoch_metrics["validation"]["explicit"]
                score = (outcome["exact_set_match"], outcome["f1"])
                if best is None or score > best["score"]:
                    best = {"epoch": epoch + 1, "checkpoint": epoch_checkpoint, "metrics": epoch_metrics,
                            "score": score}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                event("checkpoint_scored", epoch=epoch + 1, path=epoch_checkpoint,
                      exact_set_match=score[0], f1=score[1], selected_epoch=best["epoch"])
            stop = requested_stop(run / "stop-request.json", manifest["run_id"])
            if stop is not None:
                event("stop_requested_at_checkpoint", epoch=epoch + 1, path=state)
                break
            if every_epoch and stale_epochs >= config["training"].get("early_stop_patience", config["training"]["epochs"]):
                event("stopped_after_no_validation_gain", epoch=epoch + 1, selected_epoch=best["epoch"])
                break
        checkpoint = best["checkpoint"] if best is not None else adapter.checkpoint("final", sampler=True)
        record["sampler_checkpoint"] = checkpoint
        event("sampler_checkpoint_saved", path=checkpoint)
        # Retain before evaluation so an inference failure cannot discard training.
        record["retained_model"] = adapter.retain_checkpoint(checkpoint, config["run_id"])
        event("experimental_adapter_retained", model=record["retained_model"])
        if best is None:
            adapter.use_sampler(checkpoint)
            tuned = sample_phase("tuned")
        else:
            tuned = best["metrics"]
            for suffix in ("predictions", "metrics"):
                write_json(run / f"tuned-{suffix}.json", json.loads(
                    (run / f"epoch{best['epoch']}-{suffix}.json").read_text(encoding="utf-8")))
            record["selected_epoch"] = best["epoch"]
            event("best_checkpoint_selected", epoch=best["epoch"], path=checkpoint)
        price = config["pricing"]
        train_tokens = sum(e["metrics"].get("total_tokens:sum", e["training_tokens"])
                           for e in record["events"] if e["stage"] == "optimizer_step")
        sample_phases = ["base", *([f"epoch{n}" for n in range(1, epoch + 2)] if every_epoch else ["tuned"])]
        sampling_usage = {key: sum(json.loads((run / f"{phase}-metrics.json").read_text(encoding="utf-8"))[cohort]["usage"][key]
                                   for phase in sample_phases for cohort in ("validation", "train_diagnostic"))
                          for key in ("prompt_tokens", "completion_tokens")}
        estimated_cost = (train_tokens * price["train"] + sampling_usage["prompt_tokens"] * price["prefill"]
                          + sampling_usage["completion_tokens"] * price["sample"]) / 1_000_000
        comparison = {
            "base": baseline["validation"]["explicit"], "tuned": tuned["validation"]["explicit"],
            "delta": {key: tuned["validation"]["explicit"][key] - baseline["validation"]["explicit"][key]
                      for key in ("exact_set_match", "precision", "recall", "f1")},
            "base_invalid_rate": baseline["validation"]["invalid_output_rate"],
            "tuned_invalid_rate": tuned["validation"]["invalid_output_rate"],
            "completed_training_tokens": train_tokens, "all_sampling_usage": sampling_usage,
            "selected_epoch": best["epoch"] if best is not None else epoch + 1,
            "completed_epochs": epoch + 1,
            "estimated_token_cost_usd": estimated_cost, "billing_verified": False,
            "quality_promotion": False,
            "scope": "Paired descriptive smoke evaluation; small balanced official-validation subset, no held-out test or significance claim.",
        }
        write_json(run / "comparison.json", comparison)
        record["status"] = "completed"
        event("completed", comparison=comparison)
    except Exception as error:
        record["status"] = "failed"
        # SDK exceptions may contain credentials or complete request bodies.
        event("failed", error_type=type(error).__name__)
        raise
    finally:
        try:
            adapter.close()
            event("session_closed")
        finally:
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(record_path, record)
    return record
