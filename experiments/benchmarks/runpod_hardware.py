"""Bounded hardware diagnostic using an unchanged, digest-pinned trainer image.

The harness has its own Git/hash pin. The prepared training manifest retains the
original trainer commit so external checkpoints pass the normal resume checks.
This is a timing/recovery experiment, never an epoch or model-promotion run.
"""
from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import statistics

from edugraph_classify.checkpoint_storage import publish_snapshot, unpack_verified
from edugraph_classify.dataset import file_sha256
from edugraph_classify.learning_data import safe_relative, verify_prepared, write_json
from edugraph_classify.learning_run import render_prepared
from edugraph_classify.ontology import OntologyCatalog
from edugraph_classify.qwen_rendering import QwenVisionRenderer
from edugraph_classify.self_hosted_run import verify_resume


def verified_spec(run, manifest):
    pin = manifest["hardware_benchmark"]
    path = run / safe_relative(pin["spec_path"])
    if file_sha256(path) != pin["spec_sha256"]:
        raise ValueError("benchmark specification changed")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec["training_steps"] != 20 or spec["warmup_training_steps"] != 2:
        raise ValueError("benchmark is bounded to twenty training steps")
    if len(spec["generation_ids"]) != 20 or len(set(spec["generation_ids"])) != 20:
        raise ValueError("benchmark requires twenty unique generation examples")
    if ([r["id"] for r in spec["a40_generation"]] != spec["generation_ids"] or
            [r["step"] for r in spec["a40_training"]] != list(range(1, 21))):
        raise ValueError("A40 reference timings differ from the benchmark order")
    if len(spec["memory_examples"]) != 2 or any(r["split"] != "train" for r in spec["memory_examples"]):
        raise ValueError("memory probes must be two training examples")
    for row in spec["memory_examples"]:
        image = run / safe_relative(row["image_path"])
        if file_sha256(image) != row["image_sha256"]:
            raise ValueError("memory-probe image changed")
    return spec


def ordered_batches(examples, training, epoch, steps):
    order = [row for row in examples if row["split"] == "train"]
    random.Random(training["seed"] + epoch).shuffle(order)
    size = training["batch_size"]
    if size * steps > len(order):
        raise ValueError("benchmark cannot wrap an epoch")
    return [order[offset:offset + size] for offset in range(0, steps * size, size)]


def paired_summary(measured, baseline, key):
    if not measured or len(measured) != len(baseline):
        raise ValueError("paired timings must have equal nonzero sizes")
    current, previous = [r[key] for r in measured], [r[key] for r in baseline]
    if any(not 0 < value < float("inf") for value in current + previous):
        raise ValueError("timings must be positive and finite")
    return {"samples": len(current), "seconds_mean": statistics.mean(current),
            "a40_seconds_mean": statistics.mean(previous),
            "speedup": sum(previous) / sum(current)}


def gpu_memory():  # Actual CUDA integration; orchestration tests inject a fake.
    import torch
    free, total = torch.cuda.mem_get_info()
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "device_used_gib": (total - free) / 2**30, "device_total_gib": total / 2**30}


def optimizer_audit(engine, expected_step):
    import torch
    states = engine.optimizer.state_dict()["state"]
    if not states or len(states) != len(engine.parameters):
        raise ValueError("incomplete restored optimizer")
    for state in states.values():
        if int(state["step"]) != expected_step:
            raise ValueError("restored optimizer step differs from checkpoint position")
        if any(not bool(torch.isfinite(v).all()) for v in state.values() if isinstance(v, torch.Tensor)):
            raise ValueError("nonfinite optimizer state")
    return {"optimizer_states": len(states), "optimizer_step": expected_step, "finite": True}


def execute_benchmark(manifest_path, code_commit, work, engine_factory, tracker_factory, store,
                      *, resume=None, stop_requested=lambda: False, runtime_details=None,
                      renderer_factory=QwenVisionRenderer.load, memory=gpu_memory,
                      audit_optimizer=optimizer_audit):
    manifest, examples = verify_prepared(manifest_path, code_commit)
    run, config = manifest_path.parent, manifest["recipe"]
    spec = verified_spec(run, manifest)
    if resume is None:
        raise ValueError("benchmark requires the external epoch-one checkpoint")
    final_state = verify_resume(manifest, resume)
    if (final_state["epoch"], final_state["step"], final_state["offset"], final_state["phase"]) != (1, 40, 0, "final"):
        raise ValueError("unexpected source checkpoint position")
    if (runtime_details or {}).get("resume_reference") != spec["final_checkpoint"]:
        raise ValueError("worker restored a different checkpoint")
    if manifest["training_data_conflicts"] or any(version(name).split("+")[0] != pin
                                                for name, pin in manifest["runtime_versions"].items()):
        raise ValueError("runtime or training data differs from the original smoke")
    catalog = OntologyCatalog.load(config["ontology_version"])
    if catalog.snapshot_sha256 != manifest["ontology_snapshot_sha256"]:
        raise ValueError("ontology differs from the original smoke")
    work.mkdir(parents=True, exist_ok=False)
    reports = work / "reports"
    reports.mkdir()
    (reports / "events").mkdir()
    renderer = renderer_factory(run / "processor", json.loads((run / "prompt.json").read_text(encoding="utf-8")))
    rendered = render_prepared(run, config, examples + spec["memory_examples"], renderer)
    schema = json.loads((run / "closed_schema.json").read_text(encoding="utf-8"))
    rows_by_id = {row["id"]: row for row in examples}
    if any(rows_by_id[value]["split"] != "validation" for value in spec["generation_ids"]):
        raise ValueError("timing generation must use the original selection cohort")
    baseline_path = work / "baseline.tar"
    store.download(spec["baseline_checkpoint"], baseline_path)
    baseline = work / "baseline"
    unpack_verified(baseline_path, spec["baseline_checkpoint"]["sha256"], baseline)
    baseline_path.unlink()
    initial = verify_resume(manifest, baseline)
    if (initial["epoch"], initial["step"], initial["offset"]) != (0, 0, 0):
        raise ValueError("training comparison must start at the same step-zero checkpoint")
    tracker = tracker_factory({**manifest, "execution_observed": runtime_details or {}, "benchmark_only": True})
    success, event_step = False, 0
    result = {"run_id": config["run_id"], "status": "running", "quality_promotion": False,
              "trainer_commit": code_commit, "hardware_benchmark": manifest["hardware_benchmark"],
              "execution": runtime_details or {}, "training": [], "generation": []}

    def record(event, values):
        nonlocal event_step
        event_step += 1
        tracker.log({event: values}, event_step)
        print(json.dumps({"event": event, **values}), flush=True)
        write_json(reports / "events" / f"{event_step:04d}.json", {"event": event, **values})
        if stop_requested():
            raise RuntimeError("benchmark interrupted")

    try:
        record("loading_model", {})
        engine = engine_factory(manifest, renderer)
        result["audit"] = engine.audit
        if engine.audit["vision_or_bridge_trainable_parameters"] != 0:
            raise ValueError("vision or bridge parameters became trainable")
        record("model_ready", {key: value for key, value in engine.audit.items() if key != "parameter_names"})
        engine.restore(baseline)
        for step, rows in enumerate(ordered_batches(examples, config["training"], 0, spec["training_steps"]), 1):
            if [row["id"] for row in rows] != spec["a40_training"][step - 1]["ids"]:
                raise ValueError("training examples differ from the paired A40 batch")
            metrics = engine.train_batch([rendered[row["id"]] for row in rows])
            result["training"].append({"step": step, "ids": [r["id"] for r in rows], **metrics})
            record("benchmark_training_step", result["training"][-1])
        warmup = spec["warmup_training_steps"]
        result["training_summary"] = paired_summary(result["training"][warmup:], spec["a40_training"][warmup:], "seconds")
        result["training_memory"] = memory()
        record("training_comparison", result["training_summary"])

        # Discard the timing updates and restore the exact trained A40 state for
        # generation and the independent one-epoch-to-two-epoch recovery check.
        engine.restore(resume)
        result["restore_audit"] = audit_optimizer(engine, 40)
        record("checkpoint_restored", result["restore_audit"])
        for example_id in spec["generation_ids"][:2]:
            engine.predict(rendered[example_id], config["evaluation"], schema)
        for example_id in spec["generation_ids"]:
            prediction = {"id": example_id, **engine.predict(rendered[example_id], config["evaluation"], schema)}
            result["generation"].append(prediction)
            record("benchmark_prediction", {"id": example_id, "latency_seconds": prediction["latency_seconds"],
                                           "completion_tokens": prediction["usage"]["completion_tokens"]})
        result["generation_summary"] = paired_summary(result["generation"], spec["a40_generation"], "latency_seconds")
        result["generation_summary"]["identical_texts"] = sum(a["text"] == b["text"] for a, b in
            zip(result["generation"], spec["a40_generation"], strict=True))
        result["generation_summary"]["completion_tokens"] = sum(p["usage"]["completion_tokens"] for p in result["generation"])
        result["generation_summary"]["a40_completion_tokens"] = sum(p["usage"]["completion_tokens"] for p in spec["a40_generation"])
        record("generation_comparison", result["generation_summary"])

        # Restore RNG as well as tensors after sampling, then take the actual next
        # batch from epoch two. Publish coherent state before memory-only probes.
        engine.restore(resume)
        next_rows = ordered_batches(examples, config["training"], 1, 1)[0]
        update = engine.train_batch([rendered[row["id"]] for row in next_rows])
        result["resume_update"] = {"ids": [r["id"] for r in next_rows], **update, **audit_optimizer(engine, 41)}
        record("resumed_optimizer_step", result["resume_update"])
        checkpoint = work / "continuation"
        checkpoint.mkdir()
        engine.save(checkpoint)
        state = {**deepcopy(final_state), "source_run_id": final_state["run_id"], "run_id": config["run_id"],
                 "epoch": 1, "offset": len(next_rows), "step": 41, "phase": "train", "stale_epochs": 0}
        write_json(checkpoint / "state.json", state)
        write_json(checkpoint / "manifest.json", manifest)
        shutil.copytree(resume / "reports", checkpoint / "reports")
        shutil.copytree(resume / "best-adapter", checkpoint / "best-adapter")
        result["continuation"] = publish_snapshot(checkpoint, work / "checkpoint.tar", store,
            config["artifacts"]["root_uri"] + "/checkpoints/" + config["run_id"], config["run_id"])
        tracker.checkpoint(result["continuation"])
        record("recovery_checkpoint_published", result["continuation"])

        memory_rows = spec["memory_examples"]
        longest_batch = [rendered[row["id"]] for row in memory_rows] * 2
        result["longest_training_probe"] = {**engine.train_batch(longest_batch),
            "examples": [{k: r[k] for k in ("id", "training_tokens", "prompt_tokens", "image_tokens")} for r in memory_rows],
            "memory": memory(), "discarded_update": True}
        record("longest_training_passed", result["longest_training_probe"])
        engine.restore(resume)
        longest = max(memory_rows, key=lambda row: row["prompt_tokens"])
        result["longest_generation_probe"] = {"id": longest["id"],
            **engine.predict(rendered[longest["id"]], config["evaluation"], schema), "memory": memory()}
        record("longest_generation_passed", {"id": longest["id"], "memory": memory()})
        result["status"] = "completed"
        write_json(reports / "benchmark.json", result)
        reference = publish_snapshot(reports, work / "report.tar", store,
            config["artifacts"]["root_uri"] + "/benchmarks/" + config["run_id"], config["run_id"])
        result["report"] = reference
        store.complete(result, config["artifacts"]["root_uri"] + "/runs/" + config["run_id"] + "/completed.json")
        tracker.checkpoint(reference)
        success = True
        return result
    finally:
        tracker.finish(success)


if __name__ == "__main__":
    from functools import partial
    from edugraph_classify import runpod_worker
    # Reuse the tested Secure/price/identity guards, secret handling, watchdog,
    # sanitized exception reporting and publish-before-terminate lifecycle.
    runpod_worker.run_worker = partial(runpod_worker.run_worker, execute=execute_benchmark)
    raise SystemExit(runpod_worker.main())
