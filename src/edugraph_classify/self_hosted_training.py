"""Single-device QLoRA mechanics, independent of the compute provider."""
from __future__ import annotations

import io
import random
import time
from pathlib import Path

import torch
from PIL import Image

from .constrained_generation import json_constraint, tokenizer_constraints
from .learning_data import write_json
from .qwen_training_policy import LANGUAGE_PREFIX, assess_language_targets


def model_batch(example, image_processor, *, training: bool):
    """Expand the image span exactly once and mask every non-assistant target."""
    ids = [*example.before_image, *([example.image_marker_id] * example.image_tokens), *example.after_image]
    types = [0] * len(example.before_image) + [1] * example.image_tokens + [0] * len(example.after_image)
    labels = [-100] * len(ids)
    if training:
        ids.extend(example.completion)
        types.extend([0] * len(example.completion))
        labels.extend(example.completion)
    with Image.open(io.BytesIO(example.image)) as image:
        pixels = image_processor(images=image.convert("RGB"), return_tensors="pt")
    if int(pixels["image_grid_thw"].prod().item()) // image_processor.merge_size ** 2 != example.image_tokens:
        raise ValueError("image processor expanded a different number of tokens")
    batch = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
             "mm_token_type_ids": torch.tensor([types]), **pixels}
    if training:
        batch["labels"] = torch.tensor([labels])
    return batch


def trainable_audit(model):
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("base_model.model." + LANGUAGE_PREFIX)
                           or not any(part in name for part in (".lora_A.", ".lora_B."))
                           for name, _ in trainable):
        raise ValueError("trainable parameters include non-language adapters or the adapter is empty")
    return {"trainable_parameters": sum(p.numel() for _, p in trainable),
            "trainable_tensors": len(trainable), "vision_or_bridge_trainable_parameters": 0,
            "parameter_names": sorted(name for name, _ in trainable)}


def load_qlora(manifest, renderer):  # pragma: no cover - actual CUDA/bitsandbytes integration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    config = manifest["recipe"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("one CUDA GPU with BF16 support is required")
    gpu = torch.cuda.get_device_properties(0)
    if gpu.total_memory / 2**30 < config["execution"]["minimum_vram_gib"] - 1:
        raise ValueError("GPU has less memory than the approved recipe")
    seed = config["training"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = AutoModelForImageTextToText.from_pretrained(config["model"]["hf_repository"],
        revision=config["model"]["hf_revision"], trust_remote_code=False, token=False,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    targets = config["training"]["expected_target_modules"]
    linear = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
    if assess_language_targets(linear, targets) != manifest["language_target_audit"]:
        raise ValueError("loaded model's language targets differ from the prepared audit")
    paths = [name for name in linear if name.rsplit(".", 1)[-1] in targets]
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=config["training"]["rank"],
        lora_alpha=config["training"]["alpha"], lora_dropout=0.0, bias="none",
        target_modules=paths, task_type="CAUSAL_LM", revision=config["model"]["hf_revision"]))
    model.config.use_cache = False
    audit = trainable_audit(model)
    audit.update(gpu_name=gpu.name, gpu_vram_gib=gpu.total_memory / 2**30,
                 torch_version=torch.__version__, cuda_version=torch.version.cuda)
    return TorchTrainer(model, renderer, config["training"], audit=audit)


class TorchTrainer:
    def __init__(self, model, renderer, training, *, audit=None, constraints=None):
        self.model, self.renderer, self.training = model, renderer, training
        self.device = next(model.parameters()).device
        self.audit = audit or trainable_audit(model)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=training["learning_rate"],
            betas=(training["beta1"], training["beta2"]), eps=training["eps"], weight_decay=training["weight_decay"])
        self.constraints = constraints if constraints is not None else tokenizer_constraints(renderer.tokenizer)

    def batch(self, example, training):
        return {key: value.to(self.device) for key, value in
                model_batch(example, self.renderer.image_processor, training=training).items()}

    def train_batch(self, examples):
        start = time.monotonic()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        supervised = sum(len(example.completion) for example in examples)
        loss_value = 0.0
        # Token-weighted accumulation, including the partial final batch.
        for example in examples:
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                loss = self.model(**self.batch(example, True), use_cache=False).loss
                scaled = loss * len(example.completion) / supervised
            if not torch.isfinite(scaled):
                raise ValueError("nonfinite training loss; optimizer step was not applied")
            scaled.backward()
            loss_value += float(scaled.detach())
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.training["max_grad_norm"], error_if_nonfinite=True)
        self.optimizer.step()
        return {"loss": loss_value, "grad_norm": float(norm), "seconds": time.monotonic() - start,
                "supervised_tokens": supervised,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if self.device.type == "cuda" else 0.0}

    def predict(self, example, evaluation, schema):
        start = time.monotonic()
        self.model.eval()
        batch = self.batch(example, False)
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            tokens = self.model.generate(**batch, max_new_tokens=evaluation["max_tokens"],
                do_sample=False, num_beams=1, use_cache=True,
                eos_token_id=self.renderer.tokenizer.eos_token_id,
                pad_token_id=self.renderer.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=json_constraint(self.constraints, schema))
        completion = tokens[0, batch["input_ids"].shape[1]:].tolist()
        return {"text": self.renderer.tokenizer.decode(completion, skip_special_tokens=True),
                "latency_seconds": time.monotonic() - start,
                "usage": {"prompt_tokens": example.prompt_tokens, "completion_tokens": len(completion)}}

    def save_adapter(self, path):
        self.model.save_pretrained(path, safe_serialization=True)

    def load_adapter(self, path):
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        set_peft_model_state_dict(self.model, load_peft_weights(str(path), device="cpu"))

    def save(self, directory):
        self.save_adapter(directory / "adapter")
        torch.save({"optimizer": self.optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                    "python_rng": random.getstate()}, directory / "optimizer.pt")
        write_json(directory / "trainable-audit.json", self.audit)

    def restore(self, directory):
        self.load_adapter(directory / "adapter")
        state = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
