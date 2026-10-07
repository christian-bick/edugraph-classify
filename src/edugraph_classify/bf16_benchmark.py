"""Score the pinned merged BF16 checkpoint on the original validation cohort.

This diagnostic deliberately uses the training renderer and constrained
Transformers generation path. It publishes a report, not a model candidate.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .checkpoint_storage import pack_files, unpack_verified
from .constrained_generation import json_constraint, tokenizer_constraints
from .dataset import file_sha256
from .evaluation import evaluate
from .gguf_export import merge_adapter, tree_sha256
from .inference_benchmark import GpuSampler
from .inference_benchmark_config import verify_inference_prepared
from .learning_data import write_json
from .ontology import OntologyCatalog
from .prepared_rendering import render_prepared
from .qwen_rendering import QwenVisionRenderer
from .self_hosted_training import model_batch


def merged_files_digest(files_sha256: dict[str, str]) -> str:
    """Identify the complete merged checkpoint independently of file order."""
    payload = json.dumps(files_sha256, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _check_stop(stop_requested) -> None:
    if stop_requested():
        raise InterruptedError("benchmark stop requested")


def _load_merged_model(merged_dir: Path):  # pragma: no cover - real CUDA integration
    import torch
    from transformers import AutoModelForImageTextToText

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("merged BF16 inference requires one BF16-capable CUDA GPU")
    model = AutoModelForImageTextToText.from_pretrained(
        merged_dir, dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False,
        token=False, attn_implementation="sdpa",
    )
    model.eval()
    return model


def predict_merged_bf16(model, renderer: QwenVisionRenderer, rendered, row: dict,
                        schema: dict, max_tokens: int, constraints, stop_requested) -> dict:
    """Use the same greedy, constrained generation as NF4 checkpoint selection."""
    import torch

    _check_stop(stop_requested)
    started = time.monotonic()
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("merged BF16 model is not fully resident on CUDA")
    batch = {key: value.to(device) for key, value in
             model_batch(rendered, renderer.image_processor, training=False).items()}
    generation_started = time.monotonic()
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        tokens = model.generate(
            **batch, max_new_tokens=max_tokens, do_sample=False, num_beams=1,
            use_cache=True, eos_token_id=renderer.tokenizer.eos_token_id,
            pad_token_id=renderer.tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=json_constraint(constraints, schema),
        )
    generation_seconds = time.monotonic() - generation_started
    completion = tokens[0, batch["input_ids"].shape[1]:].tolist()
    text = renderer.tokenizer.decode(completion, skip_special_tokens=True)
    _check_stop(stop_requested)
    return {
        "id": row["id"], "text": text, "generated_token_ids": completion,
        "latency_seconds": time.monotonic() - started,
        "generation_seconds": generation_seconds,
        "usage": {"prompt_tokens": rendered.prompt_tokens,
                  "completion_tokens": len(completion)},
        "finish_reason": "stop" if completion and completion[-1] == renderer.tokenizer.eos_token_id
                         else "length" if len(completion) >= max_tokens else "other",
        "image_sha256": row["image_sha256"],
    }


def execute_bf16_benchmark(manifest_path: Path, work: Path, store, *,
                           stop_requested=lambda: False, runtime_details: dict | None = None,
                           merger=merge_adapter, model_loader=_load_merged_model,
                           sampler_factory=GpuSampler) -> dict:
    """Merge, attest, score and publish BF16 without uploading large weights."""
    started = time.monotonic()
    manifest_path, work = Path(manifest_path), Path(work)
    prepared = manifest_path.parent
    code_commit = json.loads(manifest_path.read_text(encoding="utf-8"))["code_commit"]
    manifest, examples = verify_inference_prepared(manifest_path, code_commit)
    recipe = manifest["recipe"]
    bench = recipe["benchmark"]
    if bench.get("route") != "merged_bf16_hf" or bench["concurrency"] != [1]:
        raise ValueError("merged BF16 benchmark requires its pinned route and concurrency one")
    source_training = manifest["source_training"]
    source_model = manifest["source_model"]
    model_pin = source_training["recipe"]["model"]
    catalog = OntologyCatalog.load(source_training["recipe"]["ontology_version"])
    if catalog.snapshot_sha256 != source_training["ontology_snapshot_sha256"]:
        raise ValueError("current ontology snapshot differs from selected training run")

    work.mkdir(parents=True, exist_ok=False)
    progress = {"run_id": manifest["run_id"], "phase": "downloading_model", "settings": []}
    write_json(work / "progress.json", progress)
    archive = work / "source-model.tar"
    store.download(source_model, archive)
    if archive.stat().st_size != source_model["bytes"]:
        raise ValueError("selected source model size changed")
    source = work / "source-model"
    unpack_verified(archive, source_model["sha256"], source)
    archive.unlink()
    if file_sha256(source / "manifest.json") != source_training["manifest_sha256"]:
        raise ValueError("selected model has a different training manifest")
    selected = json.loads((source / "result.json").read_text(encoding="utf-8"))
    if (selected.get("status") != "completed" or
            selected.get("selected_epoch") != source_training["selected_epoch"]):
        raise ValueError("selected model epoch changed")
    _check_stop(stop_requested)

    prompt = json.loads((prepared / "prompt.json").read_text(encoding="utf-8"))
    schema = json.loads((prepared / "closed_schema.json").read_text(encoding="utf-8"))
    training = json.loads((prepared / "training_examples.json").read_text(encoding="utf-8"))
    renderer = QwenVisionRenderer.load(prepared / "processor", prompt)
    rendered = render_prepared(prepared, source_training["recipe"], examples, renderer)
    constraints = tokenizer_constraints(renderer.tokenizer)
    _check_stop(stop_requested)

    progress["phase"] = "merging_bf16"
    write_json(work / "progress.json", progress)
    merged_dir = work / "export" / "merged-hf"
    merge_started = time.monotonic()
    merged = merger(source, merged_dir, model_repo=model_pin["hf_repository"],
                    model_revision=model_pin["hf_revision"])
    merge_seconds = time.monotonic() - merge_started
    if Path(merged["path"]).resolve() != merged_dir.resolve():
        raise ValueError("BF16 merger returned a path outside its output directory")
    actual_hashes = tree_sha256(merged_dir)
    if not actual_hashes or actual_hashes != merged["files_sha256"]:
        raise ValueError("merged BF16 file hashes changed before benchmarking")
    actual_digest = merged_files_digest(actual_hashes)
    if actual_digest != bench["expected_merged_files_digest"]:
        raise ValueError("merged BF16 checkpoint differs from the pinned v4 merge")
    progress["merged_files_digest"] = actual_digest
    progress["merge_seconds"] = merge_seconds
    _check_stop(stop_requested)

    progress["phase"] = "loading_bf16"
    write_json(work / "progress.json", progress)
    rounds = []
    with sampler_factory() as sampler:
        load_started = time.monotonic()
        model = model_loader(merged_dir)
        model_load_seconds = time.monotonic() - load_started
        progress["phase"] = "scoring"
        write_json(work / "progress.json", progress)
        try:
            for row in examples[:bench["warmup"]]:
                predict_merged_bf16(model, renderer, rendered[row["id"]], row, schema,
                                    bench["max_tokens"], constraints, stop_requested)
            for repeat in range(1, bench["repeats"] + 1):
                _check_stop(stop_requested)
                round_started = time.monotonic()
                predictions = [predict_merged_bf16(model, renderer, rendered[row["id"]], row,
                                                   schema, bench["max_tokens"], constraints,
                                                   stop_requested) for row in examples]
                duration = time.monotonic() - round_started
                rounds.append({"repeat": repeat, "concurrency": 1,
                               "duration_seconds": duration,
                               "throughput_examples_per_second": len(examples) / duration if duration else None,
                               "predictions": predictions,
                               "metrics": evaluate(examples, predictions, training, catalog)})
                progress["settings"] = [{"concurrency": 1,
                                         "model_load_seconds": model_load_seconds,
                                         "rounds": rounds}]
                write_json(work / "progress.json", progress)
        finally:
            del model
        gpu = sampler.summary()
    _check_stop(stop_requested)

    settings = [{"concurrency": 1, "model_load_seconds": model_load_seconds,
                 "rounds": rounds}]
    progress.update(phase="publishing", settings=settings)
    write_json(work / "progress.json", progress)
    elapsed_before_publish = time.monotonic() - started
    compute_price = (runtime_details or {}).get("compute_hour_usd", recipe["pricing"]["compute_hour_usd"])
    report = {
        "kind": "merged_bf16_inference_benchmark_result_v1", "run_id": manifest["run_id"],
        "source_training": source_training, "source_model": source_model,
        "benchmark_recipe": recipe, "code_commit": manifest["code_commit"],
        "uv_lock_sha256": manifest["uv_lock_sha256"],
        "input_manifest_sha256": file_sha256(manifest_path),
        "merged_hf": {"files_sha256": actual_hashes, "files_digest": actual_digest,
                      "merge_seconds": merge_seconds},
        "runtime": runtime_details or {}, "gpu": gpu,
        "cohort": manifest["cohort"], "cohort_count": len(examples),
        "settings": settings,
        "estimated_compute_and_disk_usd_before_publish": elapsed_before_publish / 3600 *
            (compute_price + recipe["pricing"]["disk_hour_usd"]),
        "elapsed_seconds_before_publish": elapsed_before_publish,
        "quality_promotion": False,
    }
    report_dir = work / "report"
    report_dir.mkdir()
    write_json(report_dir / "result.json", report)
    report_archive = work / "report.tar"
    pack_files(report_dir, ["result.json"], report_archive)
    root_uri = recipe["artifacts"]["root_uri"]
    report_ref = store.publish(report_archive, root_uri + "/benchmarks/" + manifest["run_id"],
                               manifest["run_id"])
    completion = {"status": "completed", "run_id": manifest["run_id"], "report": report_ref,
                  "input_manifest_sha256": report["input_manifest_sha256"],
                  "elapsed_seconds_total": time.monotonic() - started}
    completion["estimated_compute_and_disk_usd_total"] = (
        completion["elapsed_seconds_total"] / 3600 *
        (compute_price + recipe["pricing"]["disk_hour_usd"]))
    _check_stop(stop_requested)
    store.complete(completion, root_uri + "/benchmarks/" + manifest["run_id"] + "/completed.json")
    write_json(work / "completed.json", completion)
    return completion
