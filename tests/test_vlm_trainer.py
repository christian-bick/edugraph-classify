from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from edugraph_classify import vlm_trainer
from edugraph_classify.vlm_trainer import (
    QloraRecipe,
    RuntimeConfig,
    RuntimeSplit,
    TrainerConfigError,
    TrainerRuntimeError,
    load_runtime_config,
    load_training_records,
    normalize_chat_record,
)


def _record() -> dict[str, object]:
    stream = BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(stream, format="PNG")
    image = base64.b64encode(stream.getvalue()).decode("ascii")
    return {
        "messages": [
            {"role": "system", "content": "Classify."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Return JSON."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": '{"areas":["a"],"scopes":[],"abilities":["b"]}',
            },
        ]
    }


def _recipe() -> dict[str, object]:
    return {
        "method": "supervised_qlora",
        "epochs": 1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 0.00002,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "quantization": "nf4",
        "precision": "bfloat16",
        "seed": 42,
        "chat_template_thinking": False,
        "wandb": False,
    }


def _runtime(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(_record(), separators=(",", ":")) + "\n", encoding="utf-8")
        records[split] = {
            "name": split,
            "uri": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
            "example_count": 1,
        }
    payload: dict[str, object] = {
        "runtime_manifest_version": 1,
        "run_id": "vertex-smoke",
        "code_commit": "a" * 40,
        "prepared_manifest": {},
        "dataset": {"repository": "owner/data", "revision": "b" * 40, "splits": records},
        "ontology": {},
        "prompt": {},
        "schema": {},
        "model": {
            "repository": "Qwen/Qwen3.5-4B",
            "revision": "c" * 40,
            "created_at": "2026-02-27T14:45:03Z",
            "capabilities": ["image", "text", "thinking"],
        },
        "training": _recipe(),
        "provider": {},
        "output_uri": "gs://bucket-1/runs/vertex-smoke",
    }
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def test_runtime_config_and_validate_only_run_are_fully_pinned(tmp_path: Path) -> None:
    manifest, _ = _runtime(tmp_path)
    config = load_runtime_config(str(manifest))
    result = vlm_trainer.run(str(manifest), tmp_path / "work", validate_only=True)

    assert config.model_revision == "c" * 40
    assert config.recipe.chat_template_thinking is False
    assert result["status"] == "validated"
    assert result["train_examples"] == result["validation_examples"] == 1
    assert not (tmp_path / "work").exists()


def test_normalization_decodes_one_image_and_validates_dimension_target() -> None:
    normalized = normalize_chat_record(_record())
    user = normalized["messages"][1]  # type: ignore[index]
    image = user["content"][1]["image"]  # type: ignore[index]
    assert image.mode == "RGB"
    assert image.size == (2, 2)

    invalid = _record()
    invalid["messages"][2]["content"] = "not-json"  # type: ignore[index]
    with pytest.raises(TrainerConfigError, match="dimension-aware"):
        normalize_chat_record(invalid)

    missing = _record()
    missing["messages"][1]["content"] = [  # type: ignore[index]
        {"type": "text", "text": "no image"}
    ]
    with pytest.raises(TrainerConfigError, match="exactly one image"):
        normalize_chat_record(missing)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("method", "full", "supervised_qlora"),
        ("quantization", "none", "NF4"),
        ("gradient_checkpointing", False, "gradient checkpointing"),
        ("lora_rank", 3, "lora_rank"),
        ("lora_dropout", 1.0, "lora_dropout"),
        ("learning_rate", 0.0, "learning_rate"),
        ("chat_template_thinking", "false", "chat_template_thinking"),
    ],
)
def test_qlora_recipe_fails_closed(field: str, value: object, message: str) -> None:
    payload = _recipe()
    payload[field] = value
    with pytest.raises(TrainerConfigError, match=message):
        QloraRecipe.from_mapping(payload)


def test_runtime_and_split_contracts_reject_drift(tmp_path: Path) -> None:
    _, payload = _runtime(tmp_path)
    payload["model"]["revision"] = "main"  # type: ignore[index]
    with pytest.raises(TrainerConfigError, match="immutable"):
        RuntimeConfig.from_mapping(payload)

    _, payload = _runtime(tmp_path)
    payload["dataset"]["splits"]["train"]["name"] = "wrong"  # type: ignore[index]
    with pytest.raises(TrainerConfigError, match="match its key"):
        RuntimeConfig.from_mapping(payload)

    with pytest.raises(TrainerConfigError, match="size"):
        RuntimeSplit.from_mapping(
            {"name": "train", "uri": "x", "sha256": "d" * 64, "size": 0, "example_count": 1},
            "train",
        )


def test_runtime_contract_rejects_structural_and_scalar_drift(tmp_path: Path) -> None:
    _, baseline = _runtime(tmp_path)

    def check(mutate: object, message: str) -> None:
        payload = deepcopy(baseline)
        mutate(payload)  # type: ignore[operator]
        with pytest.raises((TrainerConfigError, vlm_trainer.VertexArtifactError), match=message):
            RuntimeConfig.from_mapping(payload)

    cases = [
        (lambda value: value.update(extra=True), "exactly"),
        (lambda value: value.update(runtime_manifest_version=2), "version"),
        (lambda value: value.update(run_id=""), "non-empty"),
        (lambda value: value.update(code_commit="bad"), "code_commit"),
        (lambda value: value.update(model=[]), "model must be an object"),
        (lambda value: value["model"].update(capabilities=["text"]), "capabilities"),
        (lambda value: value["dataset"].update(extra=True), "dataset must contain"),
        (lambda value: value["dataset"].update(splits=[]), "splits must be an object"),
        (lambda value: value.update(output_uri="not-gcs"), "valid gs"),
        (
            lambda value: value["dataset"]["splits"]["train"].update(sha256="bad"),
            "sha256",
        ),
        (
            lambda value: value["dataset"]["splits"]["train"].update(example_count=0),
            "example_count",
        ),
    ]
    for mutate, message in cases:
        check(mutate, message)

    recipe = _recipe()
    recipe["epochs"] = False
    with pytest.raises(TrainerConfigError, match="epochs"):
        QloraRecipe.from_mapping(recipe)


def test_normalization_rejects_each_malformed_chat_boundary() -> None:
    baseline = _record()

    def check(mutate: object, message: str) -> None:
        payload = deepcopy(baseline)
        mutate(payload)  # type: ignore[operator]
        with pytest.raises(TrainerConfigError, match=message):
            normalize_chat_record(payload)

    cases = [
        (lambda value: value.update(extra=True), "messages array"),
        (lambda value: value.update(messages=[]), "system, user"),
        (lambda value: value["messages"][0].update(role="user"), "roles"),
        (lambda value: value["messages"][0].update(content=[]), "system message"),
        (lambda value: value["messages"][2].update(content=[]), "assistant message"),
        (lambda value: value["messages"][2].update(content="[]"), "dimension-aware"),
        (lambda value: value["messages"][1].update(content="bad"), "user message"),
        (lambda value: value["messages"][1]["content"].append("bad"), "content item"),
        (
            lambda value: value["messages"][1]["content"][0].update(type="audio"),
            "unsupported",
        ),
        (
            lambda value: value["messages"][1]["content"][1].update(image_url=[]),
            "contain exactly url",
        ),
        (
            lambda value: value["messages"][1]["content"][1]["image_url"].update(url=1),
            "must be a string",
        ),
        (
            lambda value: value["messages"][1]["content"][1]["image_url"].update(
                url="https://example.com/image.png"
            ),
            "data URIs",
        ),
        (
            lambda value: value["messages"][1]["content"][1]["image_url"].update(
                url="data:image/png;base64,eA=="
            ),
            "cannot be decoded",
        ),
    ]
    for mutate, message in cases:
        check(mutate, message)


def test_training_records_verify_hash_count_and_json(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(load_training_records(RuntimeSplit(str(path), digest, 1))) == 1

    with pytest.raises(TrainerConfigError, match="sha256"):
        load_training_records(RuntimeSplit(str(path), "f" * 64, 1))
    with pytest.raises(TrainerConfigError, match="count"):
        load_training_records(RuntimeSplit(str(path), digest, 2))

    path.write_text("{\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(TrainerConfigError, match="line 1"):
        load_training_records(RuntimeSplit(str(path), digest, 1))

    path.write_bytes(b"\xff")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(TrainerConfigError, match="UTF-8"):
        load_training_records(RuntimeSplit(str(path), digest, 1))

    path.write_text("[]\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(TrainerConfigError, match="record must be an object"):
        load_training_records(RuntimeSplit(str(path), digest, 1))


def test_prompt_lengths_require_matching_multimodal_expansion_and_right_padding() -> None:
    full_ids = [[10, 20, 77, 77, 30, 40], [10, 77, 77, 77, 30, 40]]
    expanded_prompts = [[10, 20, 77, 77, 30, 0], [10, 77, 77, 77, 30, 0]]
    masks = [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]]

    assert vlm_trainer._validated_prompt_lengths(
        full_ids, expanded_prompts, masks
    ) == (5, 5)

    text_only_prompt = [[10, 20, 77, 30], [10, 77, 30, 0]]
    with pytest.raises(RuntimeError, match="not a prefix"):
        vlm_trainer._validated_prompt_lengths(
            full_ids, text_only_prompt, [[1, 1, 1, 1], [1, 1, 1, 0]]
        )
    with pytest.raises(RuntimeError, match="right padding"):
        vlm_trainer._validated_prompt_lengths(
            full_ids[:1], expanded_prompts[:1], [[1, 1, 0, 1, 1, 0]]
        )


def test_non_validation_run_writes_result_and_delegates_gpu_and_upload(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, _ = _runtime(tmp_path)
    observed: dict[str, object] = {}

    def train(config: object, train: object, validation: object, output: Path) -> dict[str, object]:
        observed["output"] = output
        return {"train_metrics": {"loss": 1.0}, "log_history": []}

    def upload(output: Path, uri: str, client: object) -> None:
        observed.update(upload=output, uri=uri, client=client)

    client = object()
    monkeypatch.setattr(vlm_trainer, "_train_qlora", train)
    monkeypatch.setattr(vlm_trainer, "_upload_output", upload)
    result = vlm_trainer.run(str(manifest), tmp_path / "work", client=client)

    output = tmp_path / "work" / "vertex-smoke"
    assert result["status"] == "completed"
    assert json.loads((output / "run-result.json").read_text(encoding="utf-8"))[
        "train_metrics"
    ]["loss"] == 1.0
    assert observed == {
        "output": output,
        "upload": output,
        "uri": "gs://bucket-1/runs/vertex-smoke",
        "client": client,
    }


def test_gcs_reader_and_process_boundary_use_safe_results(tmp_path: Path, capsys) -> None:
    manifest, _ = _runtime(tmp_path)

    class Blob:
        def download_as_bytes(self) -> bytes:
            return manifest.read_bytes()

    class Bucket:
        def blob(self, name: str) -> Blob:
            assert name == "runtime.json"
            return Blob()

    class Client:
        def bucket(self, name: str) -> Bucket:
            assert name == "bucket-1"
            return Bucket()

    assert load_runtime_config("gs://bucket-1/runtime.json", Client()).run_id == "vertex-smoke"
    assert vlm_trainer.main(["--runtime-manifest", str(manifest), "--validate-only"]) == 0
    assert '"status": "validated"' in capsys.readouterr().out

    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    assert vlm_trainer.main(["--runtime-manifest", str(bad), "--validate-only"]) == 2
    assert "must be an object" in capsys.readouterr().err


def test_runtime_stage_reports_boundary_without_exception_message(capsys) -> None:
    def fail() -> None:
        raise RuntimeError("secret-value-must-not-be-printed")

    with pytest.raises(
        TrainerRuntimeError, match=r"training stage model_load failed \(RuntimeError\)"
    ):
        vlm_trainer._run_stage("model_load", fail)

    error = capsys.readouterr().err
    events = [json.loads(line) for line in error.splitlines()]
    assert events == [
        {"event": "training_stage", "stage": "model_load", "status": "started"},
        {
            "error_type": "RuntimeError",
            "event": "training_stage",
            "stage": "model_load",
            "status": "failed",
        },
    ]
    assert "secret-value" not in error


def test_cuda_preflight_reports_only_compatibility_facts(capsys) -> None:
    class Properties:
        name = "NVIDIA A100-SXM4-40GB"
        major = 8
        minor = 0
        total_memory = 40_000_000_000

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_properties(index: int) -> Properties:
            assert index == 0
            return Properties()

    class Version:
        cuda = "12.6"

    class Torch:
        __version__ = "2.13.0+cu126"
        version = Version()
        cuda = Cuda()

    diagnostics = vlm_trainer._cuda_preflight(Torch())

    assert diagnostics == {
        "torch_version": "2.13.0+cu126",
        "torch_cuda_version": "12.6",
        "cuda_available": True,
        "cuda_device_count": 1,
        "device_name": "NVIDIA A100-SXM4-40GB",
        "compute_capability": "8.0",
        "device_memory_bytes": 40_000_000_000,
    }
    assert json.loads(capsys.readouterr().err) == {"event": "cuda_preflight", **diagnostics}


def test_cuda_preflight_fails_safely_when_device_is_unavailable(capsys) -> None:
    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class Version:
        cuda = "12.6"

    class Torch:
        __version__ = "2.13.0+cu126"
        version = Version()
        cuda = Cuda()

    with pytest.raises(TrainerRuntimeError, match="CUDA preflight failed"):
        vlm_trainer._cuda_preflight(Torch())

    assert json.loads(capsys.readouterr().err) == {
        "cuda_available": False,
        "cuda_device_count": 0,
        "event": "cuda_preflight",
        "torch_cuda_version": "12.6",
        "torch_version": "2.13.0+cu126",
    }


def test_process_boundary_preserves_safe_runtime_stage(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest, _ = _runtime(tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        raise TrainerRuntimeError("training stage model_load failed (RuntimeError)")

    monkeypatch.setattr(vlm_trainer, "run", fail)
    assert vlm_trainer.main(["--runtime-manifest", str(manifest)]) == 1
    assert capsys.readouterr().err == "training stage model_load failed (RuntimeError)\n"
