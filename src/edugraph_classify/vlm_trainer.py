"""Provider-neutral QLoRA entry point for one-image classification records."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, TypeVar

from PIL import Image as PillowImage

from .contracts import ContractError, LabelSet
from .vertex_artifacts import VertexArtifactError, split_gcs_uri


class TrainerConfigError(ValueError):
    """Raised when the immutable runtime manifest or its data is invalid."""


class TrainerRuntimeError(RuntimeError):
    """Raised with a secret-safe stage name when the training runtime fails."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DATA_URI = re.compile(r"^data:(image/[a-z0-9.+-]+);base64,([A-Za-z0-9+/]*={0,2})$")
_T = TypeVar("_T")


def _emit_runtime_event(event: str, **details: object) -> None:
    print(
        json.dumps({"event": event, **details}, ensure_ascii=True, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _run_stage(name: str, operation: Callable[[], _T]) -> _T:
    """Run one named operation without exposing third-party exception messages."""

    _emit_runtime_event("training_stage", stage=name, status="started")
    try:
        result = operation()
    except TrainerRuntimeError:
        raise
    except Exception as error:
        error_type = type(error).__name__
        _emit_runtime_event(
            "training_stage", stage=name, status="failed", error_type=error_type
        )
        raise TrainerRuntimeError(f"training stage {name} failed ({error_type})") from error
    _emit_runtime_event("training_stage", stage=name, status="completed")
    return result


def _cuda_preflight(torch: Any) -> Mapping[str, object]:
    """Emit non-secret CUDA compatibility facts and fail before model loading."""

    available = bool(torch.cuda.is_available())
    diagnostics: dict[str, object] = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": available,
        "cuda_device_count": int(torch.cuda.device_count()) if available else 0,
    }
    if available:
        properties = torch.cuda.get_device_properties(0)
        diagnostics.update(
            device_name=str(properties.name),
            compute_capability=f"{properties.major}.{properties.minor}",
            device_memory_bytes=int(properties.total_memory),
        )
    _emit_runtime_event("cuda_preflight", **diagnostics)
    if not available:
        raise TrainerRuntimeError("CUDA preflight failed (device unavailable)")
    return diagnostics


def _exact(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise TrainerConfigError(f"{name} must contain exactly {sorted(expected)}")


def _mapping(payload: Mapping[str, object], key: str, name: str = "runtime manifest") -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TrainerConfigError(f"{name}.{key} must be an object")
    return value


def _text(payload: Mapping[str, object], key: str, name: str = "runtime manifest") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise TrainerConfigError(f"{name}.{key} must be a non-empty string")
    return value


def _integer(payload: Mapping[str, object], key: str, minimum: int = 1) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise TrainerConfigError(f"training.{key} must be an integer of at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeSplit:
    uri: str
    sha256: str
    example_count: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object], name: str) -> RuntimeSplit:
        expected = {"name", "uri", "sha256", "size", "example_count"}
        _exact(payload, expected, f"dataset.splits.{name}")
        if payload.get("name") != name:
            raise TrainerConfigError(f"dataset.splits.{name}.name must match its key")
        uri = _text(payload, "uri", f"dataset.splits.{name}")
        sha256 = _text(payload, "sha256", f"dataset.splits.{name}")
        if not _SHA256.fullmatch(sha256):
            raise TrainerConfigError(f"dataset.splits.{name}.sha256 is invalid")
        count = payload.get("example_count")
        size = payload.get("size")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise TrainerConfigError(f"dataset.splits.{name}.example_count is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise TrainerConfigError(f"dataset.splits.{name}.size is invalid")
        return cls(uri=uri, sha256=sha256, example_count=count)


@dataclass(frozen=True, slots=True)
class QloraRecipe:
    epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    seed: int
    chat_template_thinking: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> QloraRecipe:
        expected = {
            "method",
            "epochs",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "learning_rate",
            "train_batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "quantization",
            "precision",
            "seed",
            "chat_template_thinking",
            "wandb",
        }
        _exact(payload, expected, "training")
        if payload.get("method") != "supervised_qlora":
            raise TrainerConfigError("training.method must be supervised_qlora")
        if payload.get("quantization") != "nf4" or payload.get("precision") != "bfloat16":
            raise TrainerConfigError("the first Vertex recipe requires NF4 and bfloat16")
        if payload.get("gradient_checkpointing") is not True or payload.get("wandb") is not False:
            raise TrainerConfigError("gradient checkpointing must be enabled and W&B disabled")
        dropout = payload.get("lora_dropout")
        learning_rate = payload.get("learning_rate")
        thinking = payload.get("chat_template_thinking")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
            raise TrainerConfigError("training.lora_dropout must be in [0, 1)")
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or learning_rate <= 0
        ):
            raise TrainerConfigError("training.learning_rate must be positive")
        if not isinstance(thinking, bool):
            raise TrainerConfigError("training.chat_template_thinking must be a boolean")
        rank = _integer(payload, "lora_rank")
        if rank not in {4, 8, 16, 32}:
            raise TrainerConfigError("training.lora_rank must be 4, 8, 16, or 32")
        return cls(
            epochs=_integer(payload, "epochs"),
            lora_rank=rank,
            lora_alpha=_integer(payload, "lora_alpha"),
            lora_dropout=float(dropout),
            learning_rate=float(learning_rate),
            train_batch_size=_integer(payload, "train_batch_size"),
            eval_batch_size=_integer(payload, "eval_batch_size"),
            gradient_accumulation_steps=_integer(payload, "gradient_accumulation_steps"),
            seed=_integer(payload, "seed", minimum=0),
            chat_template_thinking=thinking,
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    run_id: str
    code_commit: str
    model_repository: str
    model_revision: str
    train: RuntimeSplit
    validation: RuntimeSplit
    recipe: QloraRecipe
    output_uri: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> RuntimeConfig:
        expected = {
            "runtime_manifest_version",
            "run_id",
            "code_commit",
            "prepared_manifest",
            "dataset",
            "ontology",
            "prompt",
            "schema",
            "model",
            "training",
            "provider",
            "output_uri",
        }
        _exact(payload, expected, "runtime manifest")
        if payload.get("runtime_manifest_version") != 1:
            raise TrainerConfigError("runtime manifest version must be 1")
        run_id = _text(payload, "run_id")
        commit = _text(payload, "code_commit")
        if not _COMMIT.fullmatch(commit):
            raise TrainerConfigError("runtime manifest code_commit is invalid")
        model = _mapping(payload, "model")
        _exact(model, {"repository", "revision", "created_at", "capabilities"}, "model")
        revision = _text(model, "revision", "model")
        if not _COMMIT.fullmatch(revision):
            raise TrainerConfigError("model.revision must be an immutable Git commit")
        if model.get("capabilities") != ["image", "text", "thinking"]:
            raise TrainerConfigError("model capabilities do not satisfy this VLM run")
        dataset = _mapping(payload, "dataset")
        _exact(dataset, {"repository", "revision", "splits"}, "dataset")
        splits = _mapping(dataset, "splits", "dataset")
        _exact(splits, {"train", "validation"}, "dataset.splits")
        output_uri = _text(payload, "output_uri")
        split_gcs_uri(output_uri)
        return cls(
            run_id=run_id,
            code_commit=commit,
            model_repository=_text(model, "repository", "model"),
            model_revision=revision,
            train=RuntimeSplit.from_mapping(_mapping(splits, "train", "dataset.splits"), "train"),
            validation=RuntimeSplit.from_mapping(
                _mapping(splits, "validation", "dataset.splits"), "validation"
            ),
            recipe=QloraRecipe.from_mapping(_mapping(payload, "training")),
            output_uri=output_uri,
        )


def _read_uri(uri: str, client: object | None = None) -> bytes:
    if uri.startswith("gs://"):
        if client is None:
            from google.cloud import storage

            client = storage.Client()
        bucket_name, object_name = split_gcs_uri(uri)
        return client.bucket(bucket_name).blob(object_name).download_as_bytes()  # type: ignore[attr-defined,no-any-return]
    return Path(uri).read_bytes()


def load_runtime_config(uri: str, client: object | None = None) -> RuntimeConfig:
    content = _read_uri(uri, client)
    payload = json.loads(content)
    if not isinstance(payload, Mapping):
        raise TrainerConfigError("runtime manifest must be an object")
    return RuntimeConfig.from_mapping(payload)


def _decode_image_url(value: object) -> PillowImage.Image:
    if not isinstance(value, Mapping) or set(value) != {"url"}:
        raise TrainerConfigError("image_url content must contain exactly url")
    url = value["url"]
    if not isinstance(url, str):
        raise TrainerConfigError("image_url.url must be a string")
    match = _DATA_URI.fullmatch(url)
    if match is None:
        raise TrainerConfigError("training images must be embedded base64 data URIs")
    try:
        content = base64.b64decode(match.group(2), validate=True)
        with PillowImage.open(BytesIO(content)) as image:
            image.load()
            return image.convert("RGB")
    except Exception as error:
        raise TrainerConfigError("training image data cannot be decoded") from error


def normalize_chat_record(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate one provider JSONL row and convert it to a Transformers VLM chat."""

    if set(payload) != {"messages"} or not isinstance(payload["messages"], list):
        raise TrainerConfigError("training record must contain exactly a messages array")
    messages = payload["messages"]
    if len(messages) != 3 or any(not isinstance(message, Mapping) for message in messages):
        raise TrainerConfigError("training record must contain system, user, and assistant messages")
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise TrainerConfigError("training record roles must be system, user, assistant")
    system, user, assistant = messages
    if set(system) != {"role", "content"} or not isinstance(system["content"], str):
        raise TrainerConfigError("system message is invalid")
    if set(assistant) != {"role", "content"} or not isinstance(assistant["content"], str):
        raise TrainerConfigError("assistant message is invalid")
    try:
        target = json.loads(assistant["content"])
        if not isinstance(target, Mapping):
            raise ContractError("target must be an object")
        LabelSet.from_mapping(target)
    except (json.JSONDecodeError, ContractError) as error:
        raise TrainerConfigError("assistant target is not a dimension-aware label object") from error

    if set(user) != {"role", "content"} or not isinstance(user["content"], list):
        raise TrainerConfigError("user message is invalid")
    normalized_user: list[dict[str, object]] = []
    image_count = 0
    for item in user["content"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("type"), str):
            raise TrainerConfigError("user content item is invalid")
        if item["type"] == "text" and set(item) == {"type", "text"} and isinstance(
            item.get("text"), str
        ):
            normalized_user.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image_url" and set(item) == {"type", "image_url"}:
            normalized_user.append({"type": "image", "image": _decode_image_url(item["image_url"])})
            image_count += 1
        else:
            raise TrainerConfigError("unsupported user content item")
    if image_count != 1:
        raise TrainerConfigError("each atomic training record must contain exactly one image")
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system["content"]}]},
            {"role": "user", "content": normalized_user},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant["content"]}],
            },
        ]
    }


def load_training_records(
    spec: RuntimeSplit, client: object | None = None
) -> list[dict[str, object]]:
    content = _read_uri(spec.uri, client)
    if hashlib.sha256(content).hexdigest() != spec.sha256:
        raise TrainerConfigError("training split does not match its pinned sha256")
    records: list[dict[str, object]] = []
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise TrainerConfigError("training split is not valid UTF-8") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TrainerConfigError("record must be an object")
            records.append(normalize_chat_record(payload))
        except json.JSONDecodeError as error:
            raise TrainerConfigError(f"invalid JSONL record at line {line_number}") from error
    if len(records) != spec.example_count:
        raise TrainerConfigError("training split count does not match the runtime manifest")
    return records


def _validated_prompt_lengths(
    full_input_ids: Sequence[Sequence[int]],
    prompt_input_ids: Sequence[Sequence[int]],
    prompt_attention_masks: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Validate right-padded multimodal prompt prefixes and return their lengths."""

    if not (
        len(full_input_ids) == len(prompt_input_ids) == len(prompt_attention_masks)
    ):
        raise RuntimeError("training batch and prompt batch sizes do not match")
    lengths: list[int] = []
    for full_ids, prompt_ids, prompt_mask in zip(
        full_input_ids, prompt_input_ids, prompt_attention_masks, strict=True
    ):
        if len(prompt_ids) != len(prompt_mask):
            raise RuntimeError("prompt input IDs and attention mask sizes do not match")
        prompt_length = sum(prompt_mask)
        expected_mask = [1] * prompt_length + [0] * (len(prompt_mask) - prompt_length)
        if list(prompt_mask) != expected_mask:
            raise RuntimeError("prompt attention mask must use right padding")
        if list(full_ids[:prompt_length]) != list(prompt_ids[:prompt_length]):
            raise RuntimeError("chat-template prompt is not a prefix of its training record")
        lengths.append(prompt_length)
    return tuple(lengths)


def _train_qlora(  # pragma: no cover - requires a CUDA training runtime
    config: RuntimeConfig,
    train_records: list[dict[str, object]],
    validation_records: list[dict[str, object]],
    output_dir: Path,
) -> Mapping[str, object]:
    _emit_runtime_event("training_stage", stage="dependency_import", status="started")
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except Exception as error:
        error_type = type(error).__name__
        _emit_runtime_event(
            "training_stage",
            stage="dependency_import",
            status="failed",
            error_type=error_type,
        )
        raise TrainerRuntimeError(
            f"training stage dependency_import failed ({error_type})"
        ) from error
    _emit_runtime_event("training_stage", stage="dependency_import", status="completed")

    _run_stage("cuda_preflight", lambda: _cuda_preflight(torch))

    class Records(Dataset):
        def __init__(self, values: list[dict[str, object]]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> dict[str, object]:
            return self.values[index]

    processor = _run_stage(
        "processor_load",
        lambda: AutoProcessor.from_pretrained(
            config.model_repository,
            revision=config.model_revision,
        ),
    )
    processor.tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = _run_stage(
        "model_load",
        lambda: AutoModelForMultimodalLM.from_pretrained(
            config.model_repository,
            revision=config.model_revision,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        ),
    )
    model = _run_stage(
        "quantized_model_prepare",
        lambda: prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )
    model = _run_stage(
        "lora_attach",
        lambda: get_peft_model(
            model,
            LoraConfig(
                r=config.recipe.lora_rank,
                lora_alpha=config.recipe.lora_alpha,
                lora_dropout=config.recipe.lora_dropout,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        ),
    )
    model.config.use_cache = False

    def collate(examples: list[dict[str, object]]) -> Mapping[str, Any]:
        messages = [example["messages"] for example in examples]
        full_texts = [
            processor.apply_chat_template(
                item,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=config.recipe.chat_template_thinking,
            )
            for item in messages
        ]
        prompt_texts = [
            processor.apply_chat_template(
                item[:-1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=config.recipe.chat_template_thinking,
            )
            for item in messages
        ]
        images = [
            [part["image"] for message in item for part in message["content"] if part["type"] == "image"]
            for item in messages
        ]
        batch = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
        prompt_batch = processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        prompt_lengths = _validated_prompt_lengths(
            batch["input_ids"].tolist(),
            prompt_batch["input_ids"].tolist(),
            prompt_batch["attention_mask"].tolist(),
        )
        labels = batch["input_ids"].clone()
        for index, prompt_length in enumerate(prompt_lengths):
            labels[index, :prompt_length] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.recipe.epochs,
        per_device_train_batch_size=config.recipe.train_batch_size,
        per_device_eval_batch_size=config.recipe.eval_batch_size,
        gradient_accumulation_steps=config.recipe.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=config.recipe.learning_rate,
        optim="adamw_torch_fused",
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=1,
        remove_unused_columns=False,
        report_to="none",
        seed=config.recipe.seed,
        data_seed=config.recipe.seed,
    )
    trainer = _run_stage(
        "trainer_initialize",
        lambda: Trainer(
            model=model,
            args=arguments,
            train_dataset=Records(train_records),
            eval_dataset=Records(validation_records),
            data_collator=collate,
        ),
    )
    result = _run_stage("training", trainer.train)
    model_dir = output_dir / "model"
    _run_stage("model_save", lambda: trainer.save_model(str(model_dir)))
    _run_stage("processor_save", lambda: processor.save_pretrained(str(model_dir)))
    return {"train_metrics": dict(result.metrics), "log_history": trainer.state.log_history}


def _upload_output(local_dir: Path, output_uri: str, client: object | None = None) -> None:  # pragma: no cover - cloud runtime boundary
    if client is None:
        from google.cloud import storage

        client = storage.Client()
    bucket_name, prefix = split_gcs_uri(output_uri)
    bucket = client.bucket(bucket_name)  # type: ignore[attr-defined]
    for path in sorted(item for item in local_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(local_dir).as_posix()
        blob = bucket.blob(f"{prefix.rstrip('/')}/{relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        blob.metadata = {"edugraph-sha256": digest}
        blob.upload_from_filename(str(path), if_generation_match=0, checksum="crc32c")


def run(config_uri: str, work_dir: Path, *, validate_only: bool = False, client: object | None = None) -> dict[str, object]:
    config = load_runtime_config(config_uri, client)
    train_records = load_training_records(config.train, client)
    validation_records = load_training_records(config.validation, client)
    summary: dict[str, object] = {
        "status": "validated" if validate_only else "completed",
        "run_id": config.run_id,
        "code_commit": config.code_commit,
        "model": {"repository": config.model_repository, "revision": config.model_revision},
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "output_uri": config.output_uri,
    }
    if validate_only:
        return summary
    output_dir = work_dir / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _train_qlora(config, train_records, validation_records, output_dir)
    result_path = output_dir / "run-result.json"
    result_path.write_text(
        json.dumps({**summary, **metrics}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _run_stage("output_upload", lambda: _upload_output(output_dir, config.output_uri, client))
    return {**summary, **metrics}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an EduGraph VLM adapter")
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--work-dir", default="/tmp/edugraph-training")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args.runtime_manifest, Path(args.work_dir), validate_only=args.validate_only)
    except (TrainerConfigError, VertexArtifactError) as error:
        print(str(error), file=sys.stderr)
        return 2
    except TrainerRuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:  # pragma: no cover - final secret-safe process boundary
        print(f"trainer failed ({type(error).__name__})", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
