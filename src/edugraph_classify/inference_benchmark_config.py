"""Pinned, secret-free configuration for optional Runpod inference benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .dataset import file_sha256
from .gcs_artifacts import split_gcs_uri
from .learning_data import safe_relative
from .runpod_config import require_keys, secret_reference


SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
SECRET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def _positive_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_inference_recipe(config: dict) -> dict:
    """Validate the inference route independently of the training recipe."""
    require_keys(config, {"run_id", "source_model", "benchmark", "execution", "pricing", "artifacts"}, "inference recipe")
    if not isinstance(config["run_id"], str) or not RUN_ID.fullmatch(config["run_id"]):
        raise ValueError("invalid benchmark run ID")

    source = config["source_model"]
    require_keys(source, {"uri", "sha256", "bytes"}, "source model")
    split_gcs_uri(source["uri"])
    if not isinstance(source["sha256"], str) or not SHA256.fullmatch(source["sha256"]):
        raise ValueError("source model needs an immutable SHA-256")
    if type(source["bytes"]) is not int or source["bytes"] <= 0:
        raise ValueError("source model size must be a positive integer")

    bench = config["benchmark"]
    require_keys(bench, {"cohort", "max_tokens", "quantization", "concurrency", "warmup", "repeats", "llama_cpp_commit"}, "benchmark")
    if bench["cohort"] != "validation" or bench["quantization"] != "Q4_K_M":
        raise ValueError("this benchmark route requires the validation cohort and Q4_K_M")
    if not isinstance(bench["llama_cpp_commit"], str) or not COMMIT.fullmatch(bench["llama_cpp_commit"]):
        raise ValueError("llama.cpp must be pinned to a full commit")
    if type(bench["max_tokens"]) is not int or not 1 <= bench["max_tokens"] <= 2048:
        raise ValueError("invalid generated-token ceiling")
    if (not isinstance(bench["concurrency"], list) or not bench["concurrency"] or
            any(type(value) is not int or not 1 <= value <= 16 for value in bench["concurrency"]) or
            sorted(set(bench["concurrency"])) != bench["concurrency"] or bench["concurrency"][0] != 1):
        raise ValueError("concurrency must be sorted, unique and start at one")
    if type(bench["warmup"]) is not int or not 0 <= bench["warmup"] <= 20:
        raise ValueError("invalid warmup count")
    if type(bench["repeats"]) is not int or not 1 <= bench["repeats"] <= 5:
        raise ValueError("invalid benchmark repeat count")

    execution = config["execution"]
    require_keys(execution, {"cloud_type", "gpu_type", "gpu_count", "minimum_vram_gib", "minimum_ram_gib",
                             "minimum_vcpus", "container_disk_gb", "volume_gb", "max_hours",
                             "max_compute_hour_usd", "completion_action"}, "execution")
    if execution["cloud_type"] != "SECURE" or execution["gpu_count"] != 1:
        raise ValueError("inference benchmark requires one Secure GPU")
    if not isinstance(execution["gpu_type"], str) or not re.fullmatch(r"[A-Za-z0-9 ._-]{1,80}", execution["gpu_type"]):
        raise ValueError("invalid GPU type")
    for field in ("minimum_vram_gib", "minimum_ram_gib", "minimum_vcpus", "container_disk_gb", "volume_gb"):
        if type(execution[field]) is not int or execution[field] <= 0:
            raise ValueError(f"execution.{field} must be a positive integer")
    for field in ("max_hours", "max_compute_hour_usd"):
        if not _positive_number(execution[field]) or execution[field] >= 1000:
            raise ValueError(f"invalid execution.{field}")
    if execution["completion_action"] not in ("stop", "terminate"):
        raise ValueError("completion_action must be stop or terminate")

    pricing = config["pricing"]
    require_keys(pricing, {"compute_hour_usd", "disk_hour_usd", "estimated_hours", "estimated_cost_limit_usd"}, "pricing")
    if any(not _positive_number(pricing[field]) for field in pricing):
        raise ValueError("pricing values must be positive and finite")
    if pricing["estimated_hours"] > execution["max_hours"]:
        raise ValueError("estimated runtime exceeds the watchdog ceiling")
    if pricing["estimated_hours"] * (pricing["compute_hour_usd"] + pricing["disk_hour_usd"]) > pricing["estimated_cost_limit_usd"]:
        raise ValueError("estimated cost exceeds the recipe limit")

    artifacts = config["artifacts"]
    require_keys(artifacts, {"root_uri", "credentials_secret"}, "artifacts")
    root_bucket, root_key = split_gcs_uri(artifacts["root_uri"])
    source_bucket, source_key = split_gcs_uri(source["uri"])
    if (".." in root_key.split("/") or not isinstance(artifacts["credentials_secret"], str) or
            not SECRET_NAME.fullmatch(artifacts["credentials_secret"])):
        raise ValueError("invalid artifact root or credential secret name")
    if source_bucket != root_bucket or not source_key.startswith(root_key.rstrip("/") + "/models/"):
        raise ValueError("source model must be under the configured model artifact prefix")
    return config


def load_inference_recipe(path: Path) -> dict:
    return validate_inference_recipe(json.loads(path.read_text(encoding="utf-8")))


def manifest_identity(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_inference_prepared(path: Path, code_commit: str) -> tuple[dict, list[dict]]:
    """Check every staged input before upload and again inside the Pod."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "inference_benchmark_v1" or manifest.get("code_commit") != code_commit:
        raise ValueError("benchmark bundle has a different kind or code commit")
    recipe = validate_inference_recipe(manifest["recipe"])
    if manifest["run_id"] != recipe["run_id"] or manifest["source_model"] != recipe["source_model"]:
        raise ValueError("benchmark manifest differs from its recipe")
    files = manifest["files_sha256"]
    if not isinstance(files, dict) or not files or "examples.json" not in files or "training_examples.json" not in files:
        raise ValueError("benchmark input file list is incomplete")
    for name, digest in files.items():
        location = path.parent / safe_relative(name)
        if (not isinstance(digest, str) or not SHA256.fullmatch(digest) or
                not location.resolve().is_relative_to(path.parent.resolve()) or
                not location.is_file() or file_sha256(location) != digest):
            raise ValueError("benchmark input hash changed or path escaped the run directory")
    if recipe != json.loads((path.parent / "recipe.json").read_text(encoding="utf-8")):
        raise ValueError("benchmark recipe differs from its hashed copy")
    examples = json.loads((path.parent / "examples.json").read_text(encoding="utf-8"))
    if (not isinstance(examples, list) or not examples or len(examples) != manifest["cohort_count"] or
            any(row.get("split") != "validation" or row.get("image_path") not in files for row in examples) or
            len({row["id"] for row in examples}) != len(examples)):
        raise ValueError("benchmark cohort differs from the prepared validation examples")
    training = json.loads((path.parent / "training_examples.json").read_text(encoding="utf-8"))
    if not isinstance(training, list) or len(training) != manifest["training_reference_count"]:
        raise ValueError("training frequency reference differs from its manifest")
    return manifest, examples


def pod_request(manifest: dict, image: str, staged: dict) -> dict:
    """Render a Runpod request containing only public pins and secret references."""
    recipe = validate_inference_recipe(manifest["recipe"])
    if not isinstance(image, str) or not re.fullmatch(r"ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("benchmark image must be pinned to a public GHCR digest")
    if (staged.get("run_id") != recipe["run_id"] or not isinstance(staged.get("sha256"), str) or
            not SHA256.fullmatch(staged["sha256"]) or staged.get("manifest_identity") != manifest_identity(manifest)):
        raise ValueError("staged bundle identity differs from the benchmark")
    root_bucket, root_key = split_gcs_uri(recipe["artifacts"]["root_uri"])
    stage_bucket, stage_key = split_gcs_uri(staged["uri"])
    if stage_bucket != root_bucket or not stage_key.startswith(root_key.rstrip("/") + "/inputs/"):
        raise ValueError("staged input bundle is outside the configured artifact prefix")
    execution = recipe["execution"]
    return {
        "name": recipe["run_id"], "cloudType": "SECURE", "computeType": "GPU",
        "gpuCount": 1, "gpuTypeIds": [execution["gpu_type"]], "gpuTypePriority": "availability",
        "minRAMPerGPU": execution["minimum_ram_gib"], "minVCPUPerGPU": execution["minimum_vcpus"],
        "containerDiskInGb": execution["container_disk_gb"], "volumeInGb": execution["volume_gb"],
        "volumeMountPath": "/workspace", "imageName": image, "interruptible": False,
        "dockerEntrypoint": ["/app/.venv/bin/python", "-m", "edugraph_classify.inference_benchmark_worker"],
        "dockerStartCmd": [], "ports": [], "supportPublicIp": False,
        "env": {
            "EDUGRAPH_GCS_CREDENTIALS_JSON": secret_reference(recipe["artifacts"]["credentials_secret"]),
            "EDUGRAPH_BUNDLE_URI": staged["uri"], "EDUGRAPH_BUNDLE_SHA256": staged["sha256"],
            "EDUGRAPH_RUN_ID": recipe["run_id"], "EDUGRAPH_EXPECTED_COMMIT": manifest["code_commit"],
            "EDUGRAPH_MAX_HOURS": str(execution["max_hours"]), "HF_HOME": "/workspace/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        },
    }
