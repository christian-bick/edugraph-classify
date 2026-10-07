"""Validated Secure Pod settings and secret-free launch requests."""
from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path

from .learning_data import RUNTIME_PACKAGES, load_recipe
from .gcs_artifacts import split_gcs_uri

SHA256 = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def require_keys(value: dict, keys: set[str], section: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid {section} fields")


def validate_runpod_recipe(config: dict) -> dict:
    if config["model"]["provider"] != "runpod" or config["model"]["execution_mode"] != "self_hosted":
        raise ValueError("Runpod Pods require runpod/self_hosted identity")
    execution = config["execution"]
    require_keys(execution, {"cloud_type", "gpu_type", "gpu_count", "minimum_vram_gib", "minimum_ram_gib",
                            "minimum_vcpus", "container_disk_gb", "volume_gb", "max_hours",
                            "max_compute_hour_usd", "checkpoint_steps", "completion_action"}, "execution")
    if execution["cloud_type"] != "SECURE" or execution["gpu_count"] != 1:
        raise ValueError("this recipe requires a single Secure GPU")
    if not re.fullmatch(r"[A-Za-z0-9 ._-]{1,80}", execution["gpu_type"]):
        raise ValueError("invalid GPU type")
    for field in ("minimum_vram_gib", "minimum_ram_gib", "minimum_vcpus", "container_disk_gb", "volume_gb", "checkpoint_steps"):
        if type(execution[field]) is not int or execution[field] <= 0:
            raise ValueError(f"execution.{field} must be a positive integer")
    for field in ("max_hours", "max_compute_hour_usd"):
        if type(execution[field]) not in (float, int) or not 0 < execution[field] < 1000:
            raise ValueError(f"invalid execution.{field}")
    if execution["completion_action"] not in ("stop", "terminate"):
        raise ValueError("completion_action must be stop or terminate")
    if config["evaluation"].get("output_mode") != "json_schema" or config["evaluation"].get("every_epoch") is not True:
        raise ValueError("self-hosted training requires constrained generation every epoch")
    training = config["training"]
    if not training.get("expected_target_modules") or training.get("train_unembed") is not False:
        raise ValueError("explicit language-only LoRA targets and frozen output head are required")
    if training.get("train_attn") is not True or training.get("train_mlp") is not True:
        raise ValueError("the audited language attention and MLP targets must be enabled")
    for field in ("beta1", "beta2"):
        if type(training.get(field)) not in (float, int) or not 0 < training[field] < 1:
            raise ValueError(f"invalid training.{field}")
    if training.get("eps", 0) <= 0 or training.get("weight_decay", -1) < 0:
        raise ValueError("invalid AdamW policy")
    if any(training.get(k) != v for k, v in {
        "method": "nf4_qlora_bf16", "micro_batch_size": 1, "gradient_checkpointing": True,
        "lora_dropout": 0.0, "max_grad_norm": 1.0, "lr_schedule": "constant"}.items()):
        raise ValueError("unsupported QLoRA policy; update the executor before changing these settings")
    tracking = config["tracking"]
    require_keys(tracking, {"project", "entity", "api_key_secret"}, "tracking")
    if not all(isinstance(tracking[k], str) and NAME.fullmatch(tracking[k]) for k in ("project", "api_key_secret")):
        raise ValueError("tracking needs project and secret names, never secret values")
    if tracking["entity"] is not None and (not isinstance(tracking["entity"], str) or not NAME.fullmatch(tracking["entity"])):
        raise ValueError("invalid W&B entity")
    artifacts = config["artifacts"]
    require_keys(artifacts, {"root_uri", "credentials_secret"}, "artifacts")
    _, prefix = split_gcs_uri(artifacts["root_uri"])
    if ".." in prefix.split("/") or not NAME.fullmatch(artifacts["credentials_secret"]):
        raise ValueError("invalid artifact prefix or credentials secret name")
    require_keys(config["pricing"], {"compute_hour_usd", "disk_hour_usd", "estimated_hours"}, "pricing")
    if config["pricing"]["estimated_hours"] > execution["max_hours"]:
        raise ValueError("estimated runtime exceeds the maximum runtime")
    estimate = config["pricing"]["estimated_hours"] * (config["pricing"]["compute_hour_usd"] + config["pricing"]["disk_hour_usd"])
    if estimate > config["estimated_cost_limit_usd"]:
        raise ValueError("estimated cost exceeds the recipe limit")
    return config


def load_runpod_recipe(path: Path) -> dict:
    return validate_runpod_recipe(load_recipe(path, extra_fields=frozenset({"execution", "tracking", "artifacts"})))


def secret_reference(name: str) -> str:
    return "{{ RUNPOD_SECRET_" + name + " }}"


def manifest_identity(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def pod_request(manifest: dict, image: str, staged: dict, *, resume: dict | None = None, attempt: str | None = None) -> dict:
    """Render only reviewed public values and encrypted-secret references."""
    config = validate_runpod_recipe(manifest["recipe"])
    if not re.fullmatch(r"ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("the public GHCR image must be pinned by digest")
    if (staged.get("run_id") != config["run_id"] or not SHA256.fullmatch(staged.get("sha256", ""))
            or staged.get("manifest_identity") != manifest_identity(manifest)):
        raise ValueError("staged bundle identity differs from the run")
    bucket, key = split_gcs_uri(staged["uri"])
    root_bucket, root_key = split_gcs_uri(config["artifacts"]["root_uri"])
    if bucket != root_bucket or not key.startswith(root_key.rstrip("/") + "/"):
        raise ValueError("staged bundle is outside the configured artifact root")
    execution, tracking = config["execution"], config["tracking"]
    env = {"WANDB_API_KEY": secret_reference(tracking["api_key_secret"]),
           "EDUGRAPH_GCS_CREDENTIALS_JSON": secret_reference(config["artifacts"]["credentials_secret"]),
           "EDUGRAPH_BUNDLE_URI": staged["uri"], "EDUGRAPH_BUNDLE_SHA256": staged["sha256"],
           "EDUGRAPH_RUN_ID": config["run_id"], "EDUGRAPH_EXPECTED_COMMIT": manifest["code_commit"],
           "EDUGRAPH_MAX_HOURS": str(execution["max_hours"]),
           "HF_HOME": "/workspace/huggingface", "WANDB_DIR": "/workspace/wandb",
           "TORCH_DISABLE_NATIVE_JIT": "1", "TOKENIZERS_PARALLELISM": "false"}
    if resume is not None:
        split_gcs_uri(resume["uri"])
        if not SHA256.fullmatch(resume.get("sha256", "")):
            raise ValueError("resume requires an immutable snapshot checksum")
        env.update(EDUGRAPH_RESUME_URI=resume["uri"], EDUGRAPH_RESUME_SHA256=resume["sha256"])
    if attempt is not None and (resume is None or not re.fullmatch(r"[a-z0-9-]{1,16}", attempt)):
        raise ValueError("an attempt name is only valid with a resume reference and must be a short slug")
    if resume is not None and resume.get("run_id") == config["run_id"] and not attempt:
        raise ValueError("recovering the same run requires a fresh --attempt name")
    return {"name": config["run_id"] + ("--" + attempt if attempt else ""), "cloudType": execution["cloud_type"], "computeType": "GPU",
            "gpuCount": 1, "gpuTypeIds": [execution["gpu_type"]], "gpuTypePriority": "availability",
            "minRAMPerGPU": execution["minimum_ram_gib"], "minVCPUPerGPU": execution["minimum_vcpus"],
            "containerDiskInGb": execution["container_disk_gb"], "volumeInGb": execution["volume_gb"],
            "volumeMountPath": "/workspace", "imageName": image, "interruptible": False,
            "dockerEntrypoint": ["/app/.venv/bin/python", "-m", "edugraph_classify.runpod_worker"],
            "dockerStartCmd": [], "ports": [], "supportPublicIp": False, "env": env}
