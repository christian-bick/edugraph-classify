from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from edugraph_classify.checkpoint_storage import file_sha256, publish_snapshot, unpack_verified
from edugraph_classify.learning_data import write_json
from edugraph_classify.self_hosted_run import resume_identity, verify_resume
from test_learning import ROOT, renderer_factory
from test_runpod import Engine, Store, prepared


module_spec = importlib.util.spec_from_file_location("hardware_benchmark", ROOT / "experiments/benchmarks/runpod_hardware.py")
benchmark = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(benchmark)


def test_order_and_pairing_are_deterministic_and_strict():
    rows = [{"id": str(i), "split": "train"} for i in range(80)]
    training = {"seed": 42, "batch_size": 4}
    batches = benchmark.ordered_batches(rows, training, 0, 20)
    assert batches == benchmark.ordered_batches(rows, training, 0, 20)
    assert batches != benchmark.ordered_batches(rows, training, 1, 20)
    assert {r["id"] for batch in batches for r in batch} == {r["id"] for r in rows}
    with pytest.raises(ValueError, match="wrap"):
        benchmark.ordered_batches(rows, training, 0, 21)
    assert benchmark.paired_summary([{"s": 2}], [{"s": 3}], "s")["speedup"] == 1.5
    for current, prior in [([], []), ([{"s": 2}], []), ([{"s": float("nan")}], [{"s": 1}]),
                           ([{"s": 0}], [{"s": 1}])]:
        with pytest.raises(ValueError):
            benchmark.paired_summary(current, prior, "s")


def test_optimizer_audit_rejects_missing_wrong_and_nonfinite_states():
    state = {"step": torch.tensor(40), "exp_avg": torch.tensor([1.0])}
    engine = NS(parameters=[None], optimizer=NS(state_dict=lambda: {"state": {0: state}}))
    assert benchmark.optimizer_audit(engine, 40)["finite"]
    with pytest.raises(ValueError, match="step"):
        benchmark.optimizer_audit(engine, 41)
    state["exp_avg"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        benchmark.optimizer_audit(engine, 40)
    engine.parameters = [None, None]
    with pytest.raises(ValueError, match="incomplete"):
        benchmark.optimizer_audit(engine, 40)


@pytest.fixture
def case(prepared, tmp_path):
    manifest = deepcopy(prepared.manifest)
    config = manifest["recipe"]
    config["selection"].update(train=80, validation=22, final_validation=0)
    config["training"].update(batch_size=4, epochs=2)
    original = json.loads((prepared.run / "examples.json").read_text())
    rows = [{**original[0], "id": f"train-{i}", "split": "train"} for i in range(80)]
    rows += [{**original[-1], "id": f"val-{i}", "split": "validation"} for i in range(22)]
    memory_rows = [{**rows[i], "id": f"memory-{i}"} for i in range(2)]
    write_json(prepared.run / "recipe.json", config)
    write_json(prepared.run / "examples.json", rows)
    for name in ("recipe.json", "examples.json"):
        manifest["files_sha256"][name] = file_sha256(prepared.run / name)
    store = Store(tmp_path / "store")
    checkpoints = []
    for step in (0, 40):
        path = tmp_path / f"checkpoint-{step}"
        path.mkdir()
        engine = Engine()
        engine.steps = step
        engine.save(path)
        engine.save_adapter(path / "best-adapter")
        (path / "reports").mkdir()
        write_json(path / "reports/baseline-metrics.json", {"original": True})
        write_json(path / "state.json", {"identity": resume_identity(manifest), "run_id": "source",
            "epoch": int(step > 0), "step": step, "offset": 0, "phase": "final" if step else "train",
            "best": {"epoch": 1, "score": [0, 0.6]}, "stale_epochs": 0})
        ref = publish_snapshot(path, tmp_path / f"checkpoint-{step}.tar", store, "gs://bucket/checkpoints", "source")
        checkpoints.append((path, {key: ref[key] for key in ("uri", "sha256")}))
    ids = [f"val-{i}" for i in range(20)]
    spec = {"training_steps": 20, "warmup_training_steps": 2, "generation_ids": ids,
            "memory_examples": memory_rows, "baseline_checkpoint": checkpoints[0][1], "final_checkpoint": checkpoints[1][1],
            "a40_training": [{"step": i, "seconds": 2, "ids": [r["id"] for r in batch]} for i, batch in
                enumerate(benchmark.ordered_batches(rows, config["training"], 0, 20), 1)],
            "a40_generation": [{"id": value, "text": "{}", "latency_seconds": 0.2,
                                "usage": {"completion_tokens": 5}} for value in ids]}
    write_json(prepared.run / "benchmark.json", spec)
    manifest["hardware_benchmark"] = {"spec_path": "benchmark.json", "spec_sha256": file_sha256(prepared.run / "benchmark.json")}
    write_json(prepared.path, manifest)
    return NS(prepared=prepared, manifest=manifest, spec=spec, checkpoints=checkpoints, store=store, tmp=tmp_path)


class TimingEngine(Engine):
    def train_batch(self, examples):
        return {**super().train_batch(examples), "seconds": 1}


def run_case(case, engine=None, **kwargs):
    tracker, engine = Mock(), engine or TimingEngine()
    def audit(actual, expected):
        assert actual.steps == expected
        return {"optimizer_step": actual.steps}
    result = benchmark.execute_benchmark(case.prepared.path, "a" * 40, case.tmp / "work",
        lambda *a: engine, lambda *a: tracker, case.store, resume=case.checkpoints[1][0],
        runtime_details={"resume_reference": case.checkpoints[1][1]}, renderer_factory=renderer_factory,
        memory=lambda: {"peak_allocated_gib": 29}, audit_optimizer=audit, **kwargs)
    return result, tracker, engine


def test_bounded_replay_restores_state_and_excludes_memory_updates(case):
    result, tracker, engine = run_case(case)
    assert result["training_summary"] == {"samples": 18, "seconds_mean": 1, "a40_seconds_mean": 2, "speedup": 2}
    assert result["generation_summary"]["samples"] == 20
    assert result["generation_summary"]["speedup"] == pytest.approx(2)
    assert result["resume_update"]["optimizer_step"] == 41
    restored = case.tmp / "published-continuation"
    ref = result["continuation"]
    unpack_verified(case.store.root / (ref["sha256"] + ".tar"), ref["sha256"], restored)
    state = verify_resume(case.manifest, restored)
    assert (state["epoch"], state["offset"], state["step"]) == (1, 4, 41)
    assert json.loads((restored / "adapter/weights.json").read_text())["steps"] == 41
    assert engine.steps == 40  # Final generation uses the unchanged source model.
    assert len(engine.seen) == 88  # 80 replay images + 4 resume + 4 memory-only.
    assert result["quality_promotion"] is False and result["status"] == "completed"
    assert case.store.completions[-1][0]["report"] == result["report"]
    tracker.finish.assert_called_once_with(True)


def test_failure_never_publishes_completion_and_finishes_tracking(case):
    engine = TimingEngine()
    engine.train_batch = Mock(side_effect=RuntimeError("simulated GPU failure"))
    with pytest.raises(RuntimeError, match="simulated"):
        run_case(case, engine)
    assert not case.store.completions


def test_spec_and_stop_guards(case):
    spec = deepcopy(case.spec)
    spec["training_steps"] = 200
    write_json(case.prepared.run / "benchmark.json", spec)
    with pytest.raises(ValueError, match="changed"):
        benchmark.verified_spec(case.prepared.run, case.manifest)
    case.manifest["hardware_benchmark"]["spec_sha256"] = file_sha256(case.prepared.run / "benchmark.json")
    with pytest.raises(ValueError, match="bounded"):
        benchmark.verified_spec(case.prepared.run, case.manifest)
    write_json(case.prepared.run / "benchmark.json", case.spec)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_case(case, stop_requested=lambda: True)
    assert not case.store.completions
