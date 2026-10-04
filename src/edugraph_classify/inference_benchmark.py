"""Optional, bounded GGUF inference benchmark on the pinned validation cohort.

The selected adapter is merged to BF16 before the llama.cpp-specific export.
The benchmark consumes a verified prepared bundle and publishes immutable model
and report bundles before writing its completion marker.
"""

from __future__ import annotations

import base64
import io
import json
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from .checkpoint_storage import pack_files, unpack_verified
from .dataset import file_sha256
from .evaluation import evaluate
from .gguf_export import export_gguf
from .inference_benchmark_config import verify_inference_prepared
from .learning_data import write_json
from .ontology import OntologyCatalog
from .output_contract import DIMENSIONS


def jpeg_data_url(image_path: Path) -> str:
    """Match the RGB JPEG quality used by the original training renderer."""
    with Image.open(image_path) as original:
        output = io.BytesIO()
        original.convert("RGB").save(output, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def completion_request(image_url: str, user_prompt: str, schema: dict, max_tokens: int) -> dict:
    """Use llama.cpp's direct JSON-schema response format and fixed template."""
    ordered_schema = schema
    if isinstance(schema.get("properties"), dict):
        properties = schema["properties"]
        ordered_properties = {name: properties[name] for name in DIMENSIONS if name in properties}
        ordered_properties.update(properties)
        ordered_schema = {**schema, "properties": ordered_properties}
    return {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": ordered_schema},
        "stream": False,
    }


def _free_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class LlamaServer:
    """One local llama.cpp process; rebuilt for each concurrency setting."""

    def __init__(self, *, model: Path, mmproj: Path, chat_template: Path,
                 parallel: int, context_per_slot: int, ubatch_size: int, log_path: Path,
                 stop_requested=lambda: False, binary: Path = Path("/opt/llama.cpp/build/bin/llama-server"),
                 request_timeout_seconds: float = 300):
        self.model, self.mmproj, self.chat_template = model, mmproj, chat_template
        self.parallel, self.context_per_slot = parallel, context_per_slot
        self.ubatch_size = ubatch_size
        self.log_path, self.stop_requested = log_path, stop_requested
        self.binary, self.request_timeout_seconds = binary, request_timeout_seconds
        self.port = _free_local_port()
        self.process = None
        self.log = None
        self.startup_seconds = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("wb")
        command = [str(self.binary), "-m", str(self.model), "--mmproj", str(self.mmproj),
                   "--jinja", "--chat-template-file", str(self.chat_template),
                   "--ctx-size", str(self.context_per_slot * self.parallel),
                   "--ubatch-size", str(self.ubatch_size),
                   "--parallel", str(self.parallel), "--n-gpu-layers", "all",
                   "--fit", "off", "--host", "127.0.0.1", "--port", str(self.port)]
        started = time.monotonic()
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
            while time.monotonic() - started < 600:
                if self.stop_requested():
                    raise InterruptedError("benchmark stop requested during server startup")
                if self.process.poll() is not None:
                    raise RuntimeError("llama-server exited before becoming healthy")
                try:
                    with urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                        if response.status == 200:
                            self.startup_seconds = time.monotonic() - started
                            return self
                except (HTTPError, URLError, TimeoutError):
                    pass
                time.sleep(0.5)
            raise TimeoutError("llama-server did not become healthy within 600 seconds")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        if self.log is not None:
            self.log.close()

    def infer(self, payload: dict) -> dict:
        request = Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            json.dumps(payload, separators=(",", ":")).encode(),
            {"Content-Type": "application/json"}, method="POST",
        )
        with urlopen(request, timeout=self.request_timeout_seconds) as response:
            result = json.load(response)
        if not isinstance(result, dict) or not result.get("choices"):
            raise ValueError("llama-server response lacks a completion choice")
        return result


class GpuSampler:
    """Best-effort device telemetry, with no effect on scoring or success."""

    def __init__(self):
        self.samples: list[tuple[int, int, int]] = []
        self.stopped = threading.Event()
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stopped.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    check=True, timeout=5,
                )
                fields = result.stdout.splitlines()[0].split(",")
                self.samples.append(tuple(int(value.strip()) for value in fields))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self.stopped.wait(1)

    def __exit__(self, *_):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=6)

    def summary(self) -> dict:
        return {"peak_memory_mib": max((item[0] for item in self.samples), default=None),
                "total_memory_mib": max((item[1] for item in self.samples), default=None),
                "mean_gpu_utilization_percent":
                    sum(item[2] for item in self.samples) / len(self.samples) if self.samples else None,
                "sample_count": len(self.samples)}


def _predict(server, example: dict, prepared: Path, prompt: dict, schema: dict,
             max_tokens: int, stop_requested) -> dict:
    started = time.monotonic()
    if stop_requested():
        raise InterruptedError("benchmark stop requested")
    image_path = prepared / example["image_path"]
    if file_sha256(image_path) != example["image_sha256"]:
        raise ValueError("validation image hash differs from its prepared example")
    request = completion_request(jpeg_data_url(image_path), prompt["user"], schema, max_tokens)
    server_started = time.monotonic()
    response = server.infer(request)
    server_elapsed = time.monotonic() - server_started
    elapsed = time.monotonic() - started
    choice = response["choices"][0]
    content = choice["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("llama-server completion content is not text")
    return {"id": example["id"], "text": content, "latency_seconds": elapsed,
            "server_latency_seconds": server_elapsed,
            "preprocessing_seconds": server_started - started,
            "usage": response.get("usage", {}), "finish_reason": choice.get("finish_reason"),
            "raw_response": response, "image_sha256": example["image_sha256"]}


def _check_stop(stop_requested):
    if stop_requested():
        raise InterruptedError("benchmark stop requested")


def _run_setting(server, examples: list[dict], prepared: Path, prompt: dict, schema: dict,
                 max_tokens: int, concurrency: int, warmup: int, repeats: int,
                 training: list[dict], catalog: OntologyCatalog, stop_requested,
                 on_round=lambda _: None) -> list[dict]:
    for example in examples[:warmup]:
        _predict(server, example, prepared, prompt, schema, max_tokens, stop_requested)
    results = []
    for repeat in range(1, repeats + 1):
        _check_stop(stop_requested)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            predictions = list(pool.map(
                lambda row: _predict(server, row, prepared, prompt, schema, max_tokens, stop_requested),
                examples,
            ))
        elapsed = time.monotonic() - started
        metrics = evaluate(examples, predictions, training, catalog)
        results.append({"repeat": repeat, "concurrency": concurrency,
                        "duration_seconds": elapsed,
                        "throughput_examples_per_second": len(examples) / elapsed if elapsed else None,
                        "predictions": predictions, "metrics": metrics})
        on_round(results)
    return results


def execute_benchmark(manifest_path: Path, work: Path, store, *,
                      stop_requested=lambda: False, runtime_details: dict | None = None,
                      exporter=export_gguf, server_factory=LlamaServer,
                      sampler_factory=GpuSampler, llama_dir: Path = Path("/opt/llama.cpp")) -> dict:
    """Run a verified candidate and publish the result before declaring success.

    Injection points keep expensive conversion, local server and GCS outside
    unit tests. The worker enforces Pod security and price before calling this.
    """
    started = time.monotonic()
    manifest_path, work = Path(manifest_path), Path(work)
    manifest, examples = verify_inference_prepared(manifest_path,
                                                    json.loads(manifest_path.read_text(encoding="utf-8"))["code_commit"])
    recipe = manifest["recipe"]
    source_training = manifest["source_training"]
    source_model = manifest["source_model"]
    model_pin = source_training["recipe"]["model"]
    bench = recipe["benchmark"]
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
    if selected.get("selected_epoch") != source_training["selected_epoch"]:
        raise ValueError("selected model epoch changed")
    _check_stop(stop_requested)

    progress["phase"] = "exporting_gguf"
    write_json(work / "progress.json", progress)
    export = exporter(source, work / "export", model_repo=model_pin["hf_repository"],
                      model_revision=model_pin["hf_revision"], llama_dir=llama_dir,
                      quantization=bench["quantization"])
    if export["gguf"]["llama_cpp_commit"] != bench["llama_cpp_commit"]:
        raise ValueError("GGUF converter revision differs from the benchmark recipe")
    _check_stop(stop_requested)

    export_dir = work / "export"
    prompt = json.loads((manifest_path.parent / "prompt.json").read_text(encoding="utf-8"))
    schema = json.loads((manifest_path.parent / "closed_schema.json").read_text(encoding="utf-8"))
    training = json.loads((manifest_path.parent / "training_examples.json").read_text(encoding="utf-8"))
    model = Path(export["gguf"]["language_quantized"]["path"])
    mmproj = Path(export["gguf"]["mmproj_bf16"]["path"])
    if not model.is_relative_to(export_dir.resolve()) or not mmproj.is_relative_to(export_dir.resolve()):
        raise ValueError("GGUF exporter returned a path outside its output directory")
    if file_sha256(model) != export["gguf"]["language_quantized"]["sha256"] or \
            file_sha256(mmproj) != export["gguf"]["mmproj_bf16"]["sha256"]:
        raise ValueError("GGUF export hash changed before benchmarking")

    settings = []
    with sampler_factory() as sampler:
        for concurrency in bench["concurrency"]:
            _check_stop(stop_requested)
            progress.update(phase="serving", concurrency=concurrency, settings=settings)
            write_json(work / "progress.json", progress)
            with server_factory(model=model, mmproj=mmproj,
                                chat_template=export_dir / "chat_template.jinja",
                                parallel=concurrency,
                                context_per_slot=source_training["recipe"]["training"]["max_context_tokens"],
                                ubatch_size=bench["ubatch_size"],
                                log_path=work / "logs" / f"llama-c{concurrency}.log",
                                stop_requested=stop_requested) as server:
                rounds = _run_setting(server, examples, manifest_path.parent, prompt, schema,
                                      bench["max_tokens"], concurrency, bench["warmup"],
                                      bench["repeats"], training, catalog, stop_requested,
                                      on_round=lambda partial: write_json(
                                          work / "progress.json",
                                          {**progress, "settings": [*settings, {
                                              "concurrency": concurrency,
                                              "server_startup_seconds": server.startup_seconds,
                                              "rounds": partial}]},
                                      ))
                settings.append({"concurrency": concurrency,
                                 "server_startup_seconds": server.startup_seconds,
                                 "rounds": rounds})
        gpu = sampler.summary()
    _check_stop(stop_requested)
    progress.update(phase="publishing", settings=settings)
    progress.pop("concurrency", None)
    write_json(work / "progress.json", progress)

    elapsed_before_publish = time.monotonic() - started
    compute_price = (runtime_details or {}).get("compute_hour_usd", recipe["pricing"]["compute_hour_usd"])
    report = {
        "kind": "inference_benchmark_result_v1", "run_id": manifest["run_id"],
        "source_training": source_training, "source_model": source_model,
        "benchmark_recipe": recipe, "code_commit": manifest["code_commit"],
        "uv_lock_sha256": manifest["uv_lock_sha256"],
        "input_manifest_sha256": file_sha256(manifest_path),
        "export": export, "runtime": runtime_details or {}, "gpu": gpu,
        "cohort": manifest["cohort"], "cohort_count": len(examples),
        "settings": settings,
        "estimated_compute_and_disk_usd_before_publish":
            elapsed_before_publish / 3600 * (compute_price + recipe["pricing"]["disk_hour_usd"]),
        "elapsed_seconds_before_publish": elapsed_before_publish,
        "quality_promotion": False,
    }
    report_dir = work / "report"
    report_dir.mkdir()
    write_json(report_dir / "result.json", report)
    candidate = {
        "kind": "gguf_inference_candidate_v1", "run_id": manifest["run_id"],
        "source_model": source_model, "source_training_run_id": source_training["run_id"],
        "base_model": model_pin, "quantization": bench["quantization"],
        "llama_cpp_commit": bench["llama_cpp_commit"],
        "files_sha256": {name: file_sha256(export_dir / name)
                         for name in (model.name, mmproj.name, "chat_template.jinja",
                                      "prompt.json", "closed_schema.json", "manifest.json")},
        "merged_hf_files_sha256": export["merged_hf"]["files_sha256"],
    }
    write_json(export_dir / "candidate.json", candidate)
    model_archive = work / "candidate.tar"
    pack_files(export_dir, [model.name, mmproj.name, "chat_template.jinja", "prompt.json",
                            "closed_schema.json", "manifest.json", "candidate.json"], model_archive)
    root_uri = recipe["artifacts"]["root_uri"]
    candidate_ref = store.publish(model_archive, root_uri + "/inference-models/" + manifest["run_id"],
                                  manifest["run_id"])
    write_json(report_dir / "candidate-reference.json", candidate_ref)
    report_archive = work / "report.tar"
    pack_files(report_dir, ["result.json", "candidate-reference.json"], report_archive)
    report_ref = store.publish(report_archive, root_uri + "/benchmarks/" + manifest["run_id"],
                               manifest["run_id"])
    completion = {"status": "completed", "run_id": manifest["run_id"],
                  "candidate": candidate_ref, "report": report_ref,
                  "input_manifest_sha256": report["input_manifest_sha256"],
                  "elapsed_seconds_total": time.monotonic() - started}
    completion["estimated_compute_and_disk_usd_total"] = (
        completion["elapsed_seconds_total"] / 3600 *
        (compute_price + recipe["pricing"]["disk_hour_usd"]))
    _check_stop(stop_requested)
    store.complete(completion, root_uri + "/benchmarks/" + manifest["run_id"] + "/completed.json")
    write_json(work / "completed.json", completion)
    return completion
