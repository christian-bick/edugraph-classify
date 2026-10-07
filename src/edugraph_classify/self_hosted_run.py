"""Recoverable training orchestration with generated, order-independent evaluation."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from importlib.metadata import version
from pathlib import Path

from .checkpoint_storage import publish_snapshot
from .evaluation import evaluate
from .learning_data import verify_prepared, write_json
from .prepared_rendering import render_prepared
from .ontology import OntologyCatalog
from .qwen_rendering import QwenVisionRenderer


def resume_identity(manifest):
    recipe = manifest["recipe"]
    policy = {key: recipe[key] for key in ("dataset", "ontology_version", "model", "selection", "evaluation")}
    policy["training"] = {k: v for k, v in recipe["training"].items() if k not in ("epochs", "early_stop_patience")}
    policy.update({key: manifest[key] for key in ("code_commit", "uv_lock_sha256", "runtime_versions",
                   "ontology_snapshot_sha256", "language_target_audit")})
    policy["files"] = {k: v for k, v in manifest["files_sha256"].items() if k != "recipe.json"}
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


def verify_resume(manifest, snapshot):
    state = json.loads((snapshot / "state.json").read_text(encoding="utf-8"))
    if state["identity"] != resume_identity(manifest):
        raise ValueError("checkpoint is incompatible with the pinned inputs, runtime or training policy")
    if state["phase"] == "final" and state["run_id"] != manifest["run_id"] and state["epoch"] >= manifest["recipe"]["training"]["epochs"]:
        raise ValueError("continuation needs a larger total epoch target")
    if not (snapshot / "optimizer.pt").is_file():
        raise ValueError("checkpoint lacks optimizer state")
    return state


def execute_self_hosted(manifest_path: Path, code_commit: str, work: Path, engine_factory, tracker_factory,
                        store, *, resume: Path | None = None, stop_requested=lambda: False,
                        renderer_factory=QwenVisionRenderer.load, runtime_details: dict | None = None):
    manifest, examples = verify_prepared(manifest_path, code_commit)
    config, run = manifest["recipe"], manifest_path.parent
    if manifest["training_data_conflicts"]:
        raise ValueError("conflicting gold labels must be resolved before training")
    if any(version(name).split("+")[0] != pin for name, pin in manifest["runtime_versions"].items()):
        raise ValueError("installed packages differ from the prepared runtime")
    catalog = OntologyCatalog.load(config["ontology_version"])
    if catalog.snapshot_sha256 != manifest["ontology_snapshot_sha256"]:
        raise ValueError("ontology changed since preparation")
    state = verify_resume(manifest, resume) if resume else {
        "identity": resume_identity(manifest), "run_id": config["run_id"], "phase": "baseline",
        "epoch": 0, "offset": 0, "step": 0, "best": None, "stale_epochs": 0}
    recovering = resume is not None and state["run_id"] == config["run_id"]
    completion_uri = config["artifacts"]["root_uri"] + "/runs/" + config["run_id"] + "/completed.json"
    published = store.read_json(completion_uri)
    if published:
        if published.get("identity") != state["identity"]:
            raise ValueError("completed model belongs to a different training identity")
        return published  # Final publication succeeded before a later shutdown failure.
    state["source_run_id"] = state["run_id"] if resume else None
    state["run_id"] = config["run_id"]
    if state["phase"] == "final" and not recovering:
        state.update(phase="train", stale_epochs=0)
    work.mkdir(parents=True, exist_ok=False)
    reports, best = work / "reports", work / "best-adapter"
    if resume:
        shutil.copytree(resume / "reports", reports)
        if (resume / "best-adapter").exists():
            shutil.copytree(resume / "best-adapter", best)
    else:
        reports.mkdir()
    renderer = renderer_factory(run / "processor", json.loads((run / "prompt.json").read_text(encoding="utf-8")))
    rendered = render_prepared(run, config, examples, renderer)
    schema = json.loads((run / "closed_schema.json").read_text(encoding="utf-8"))
    train = [r for r in examples if r["split"] == "train"]
    selection = [("validation", [r for r in examples if r["split"] == "validation"]),
                 ("train_diagnostic", [r for r in train if r["train_diagnostic"]])]
    tracker = tracker_factory({**manifest, "execution_observed": runtime_details or {},
        "resume_from": {"run_id": state["source_run_id"], "epoch": state["epoch"], "step": state["step"]}}, resume=recovering)
    started = time.monotonic()
    success = False
    try:
        print(json.dumps({"event": "loading_model", "run_id": config["run_id"]}), flush=True)
        engine = engine_factory(manifest, renderer)
        print(json.dumps({"event": "model_ready", "trainable_parameters": engine.audit.get("trainable_parameters"),
                          "vision_or_bridge_trainable_parameters": engine.audit.get("vision_or_bridge_trainable_parameters")}), flush=True)
        write_json(reports / "runtime-audit.json", {**engine.audit, "execution": runtime_details or {}})
        if resume:
            engine.restore(resume)

        def log(values):
            hours = (time.monotonic() - started) / 3600
            rate = (runtime_details or {}).get("compute_hour_usd", config["pricing"]["compute_hour_usd"])
            tracker.log({**values, "session": {"elapsed_hours": hours,
                "estimated_compute_and_disk_usd": hours * (rate + config["pricing"]["disk_hour_usd"])}}, state["step"])

        def sample(phase, cohorts, epoch=0):
            predictions, metrics = [], {}
            for name, cohort in cohorts:
                print(json.dumps({"event": "evaluation_start", "phase": phase, "cohort": name, "images": len(cohort)}), flush=True)
                for index, row in enumerate(cohort, 1):
                    predictions.append({"id": row["id"], **engine.predict(rendered[row["id"]], config["evaluation"], schema)})
                    write_json(reports / f"{phase}-predictions.json", predictions)
                    print(json.dumps({"event": "prediction", "phase": phase, "cohort": name, "completed": index,
                                      "latency_seconds": predictions[-1].get("latency_seconds")}), flush=True)
                ids = {r["id"] for r in cohort}
                metrics[name] = evaluate(cohort, [p for p in predictions if p["id"] in ids], train, catalog)
            write_json(reports / f"{phase}-metrics.json", metrics)
            log({"evaluation": metrics, "evaluation_epoch": epoch})
            print(json.dumps({"event": "evaluation_complete", "phase": phase,
                "metrics": {name: result["explicit"] for name, result in metrics.items()}}), flush=True)
            return metrics

        def checkpoint():
            directory = work / "checkpoint"
            directory.mkdir(exist_ok=True)
            engine.save(directory)
            write_json(directory / "state.json", state)
            write_json(directory / "manifest.json", manifest)
            shutil.copytree(reports, directory / "reports", dirs_exist_ok=True)
            if best.exists():
                shutil.copytree(best, directory / "best-adapter", dirs_exist_ok=True)
            # Do not recursively embed the previous external reference.
            (directory / "external.json").unlink(missing_ok=True)
            reference = publish_snapshot(directory, work / "snapshot.tar", store,
                config["artifacts"]["root_uri"] + "/checkpoints/" + config["run_id"], config["run_id"])
            write_json(work / "latest-checkpoint.json", reference)
            tracker.checkpoint(reference)
            print(json.dumps({"event": "checkpoint_published", "step": state["step"], "reference": reference}), flush=True)
            return reference

        if state["phase"] == "baseline":
            sample("baseline", selection)
            state["phase"] = "train"
            checkpoint()
        batch_size = config["training"]["batch_size"]
        while state["epoch"] < config["training"]["epochs"] and state["phase"] != "final":
            order = list(train)
            random.Random(config["training"]["seed"] + state["epoch"]).shuffle(order)
            for offset in range(state["offset"], len(order), batch_size):
                rows = order[offset:offset + batch_size]
                metrics = engine.train_batch([rendered[row["id"]] for row in rows])
                state.update(offset=offset + len(rows), step=state["step"] + 1)
                log({"train": metrics, "epoch": state["epoch"] + state["offset"] / len(order)})
                print(json.dumps({"event": "optimizer_step", "step": state["step"], **metrics}), flush=True)
                if state["step"] % config["execution"]["checkpoint_steps"] == 0:
                    checkpoint()
                if stop_requested():
                    reference = checkpoint()
                    success = True
                    return {"status": "paused", "checkpoint": reference, "step": state["step"]}
            epoch = state["epoch"] + 1
            metrics = sample(f"epoch-{epoch}", selection, epoch)
            outcome = metrics["validation"]["explicit"]
            score = [outcome["exact_set_match"], outcome["f1"]]
            if state["best"] is None or score > state["best"]["score"]:
                engine.save_adapter(best)
                state.update(best={"epoch": epoch, "score": score}, stale_epochs=0)
            else:
                state["stale_epochs"] += 1
            state.update(epoch=epoch, offset=0)
            checkpoint()
            if state["stale_epochs"] >= config["training"]["early_stop_patience"]:
                break
        # Save current optimizer + current adapter BEFORE loading the selected adapter.
        # This keeps continuation coherent even when the best epoch was earlier.
        state["phase"] = "final"
        continuation = checkpoint()
        engine.load_adapter(best)
        final = [r for r in examples if r["split"] == "final_validation"]
        if final:
            sample("final", [("final_validation", final)], state["best"]["epoch"])
        export = work / "export"
        export.mkdir()
        engine.save_adapter(export / "adapter")
        shutil.copytree(reports, export / "reports")
        shutil.copytree(run / "processor", export / "processor")
        for name in ("prompt.json", "schema.json", "closed_schema.json", "chat_template.jinja", "manifest.json"):
            shutil.copyfile(run / name, export / name)
        result = {"status": "completed", "run_id": config["run_id"], "identity": state["identity"],
                  "selected_epoch": state["best"]["epoch"],
                  "epochs_completed": state["epoch"], "step": state["step"],
                  "continuation": continuation, "quality_promotion": False}
        baseline = json.loads((reports / "baseline-metrics.json").read_text(encoding="utf-8"))["validation"]["explicit"]
        selected = json.loads((reports / f"epoch-{state['best']['epoch']}-metrics.json").read_text(encoding="utf-8"))["validation"]["explicit"]
        result.update(baseline_validation=baseline, selected_validation=selected,
                      validation_delta={key: selected[key] - baseline[key] for key in ("exact_set_match", "f1")})
        write_json(export / "result.json", result)
        reference = publish_snapshot(export, work / "export.tar", store,
            config["artifacts"]["root_uri"] + "/models/" + config["run_id"], config["run_id"])
        result["model"] = reference
        store.complete(result, completion_uri)
        tracker.checkpoint(reference, final=True)
        write_json(work / "result.json", result)
        success = True
        return result
    finally:
        tracker.finish(success)
