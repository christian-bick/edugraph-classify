"""Local, provider-independent Qwen vision rendering with assistant-only loss.

The released tokenizer.json is loaded verbatim: rebuilding its pre-tokenizer
through AutoTokenizer changes Qwen3.8 token IDs. Images remain explicit chunks,
so shifting targets never mistakes one placeholder for the expanded image span.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_MARKER = "<|image_pad|>"


@dataclass(frozen=True)
class VisionExample:
    before_image: tuple[int, ...]
    after_image: tuple[int, ...]
    completion: tuple[int, ...]
    image: bytes
    image_tokens: int
    image_marker_id: int

    @property
    def prompt_tokens(self) -> int:
        return len(self.before_image) + self.image_tokens + len(self.after_image)

    @property
    def training_tokens(self) -> int:
        return self.prompt_tokens + len(self.completion) - 1

    def targets_and_weights(self) -> tuple[list[int], list[float]]:
        expanded = [*self.before_image, *([0] * self.image_tokens), *self.after_image, *self.completion]
        weights = [0.0] * self.prompt_tokens + [1.0] * len(self.completion)
        return expanded[1:], weights[1:]

    def sampling_tokens(self) -> list[int]:
        return [*self.before_image, self.image_marker_id, *self.after_image]

    def fingerprint(self) -> str:
        payload = [self.before_image, self.after_image, self.completion,
                   self.image_tokens, self.image_marker_id, hashlib.sha256(self.image).hexdigest()]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def fixed_system_template(template: str, system: str) -> str:
    """Embed one immutable system message; reject conflicting caller prompts.

    This is a portable base-model template export, not a Fireworks LoRA override.
    Supplying the identical system message is accepted without duplicating it.
    """
    return (
        "{%- set classifier_system = " + json.dumps(system, ensure_ascii=False) + " %}"
        "{%- if messages and messages[0].role == 'system' %}"
        "{%- if messages[0].content != classifier_system %}"
        "{{- raise_exception('Classifier system prompt is fixed.') }}{%- endif %}"
        "{%- set messages = messages[1:] %}{%- endif %}"
        "{%- set messages = [{'role': 'system', 'content': classifier_system}] + messages %}"
        "{%- set enable_thinking = false %}" + template
    )


class QwenVisionRenderer:
    def __init__(self, tokenizer, image_processor, system: str, user: str):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.system = system
        self.user = user
        marker = tokenizer.encode(IMAGE_MARKER, add_special_tokens=False)
        if len(marker) != 1:
            raise ValueError("image marker must be one special token")
        self.image_marker_id = marker[0]

    @classmethod
    def load(cls, processor_dir: Path, prompt: dict) -> QwenVisionRenderer:
        from transformers import AutoImageProcessor, TokenizersBackend

        config = json.loads((processor_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        tokenizer = TokenizersBackend.from_pretrained(
            processor_dir, local_files_only=True,
            **{key: config[key] for key in ("add_bos_token", "add_eos_token") if key in config},
        )
        processor = AutoImageProcessor.from_pretrained(processor_dir, local_files_only=True)
        return cls(tokenizer, processor, prompt["system"], prompt["user"])

    def messages(self) -> list[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": [
                {"type": "text", "text": self.user}, {"type": "image"},
            ]},
        ]

    def render(self, image_data: bytes, target: str, *, max_tokens: int, max_output: int) -> VisionExample:
        messages = self.messages()
        kwargs = {"tokenize": False, "enable_thinking": False}
        prefix = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
        full = self.tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": target}],
            add_generation_prompt=False, **kwargs,
        )
        prompt = self.tokenizer.encode(prefix, add_special_tokens=False)
        tokens = self.tokenizer.encode(full, add_special_tokens=False)
        if tokens[:len(prompt)] != prompt or len(tokens) <= len(prompt):
            raise ValueError("training and generation prefixes differ")
        if prompt.count(self.image_marker_id) != 1 or self.image_marker_id in tokens[len(prompt):]:
            raise ValueError("one independent task image is required")
        marker_index = prompt.index(self.image_marker_id)
        with Image.open(io.BytesIO(image_data)) as original:
            rgb = original.convert("RGB")
            width, height = rgb.size
            jpeg = io.BytesIO()
            rgb.save(jpeg, format="JPEG", quality=75)
        patches = self.image_processor.get_number_of_image_patches(height, width, images_kwargs={})
        image_tokens = patches // self.image_processor.merge_size ** 2
        example = VisionExample(
            tuple(prompt[:marker_index]), tuple(prompt[marker_index + 1:]),
            tuple(tokens[len(prompt):]), jpeg.getvalue(), image_tokens, self.image_marker_id,
        )
        if image_tokens < 1 or max(example.training_tokens, example.prompt_tokens + max_output) > max_tokens:
            raise ValueError("example exceeds the configured context; truncation is forbidden")
        return example

    def export_template(self) -> str:
        template = fixed_system_template(self.tokenizer.chat_template, self.system)
        normal = self.tokenizer.apply_chat_template(
            self.messages(), tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        embedded = self.tokenizer.apply_chat_template(
            self.messages()[1:], chat_template=template, tokenize=False, add_generation_prompt=True,
        )
        if normal != embedded:
            raise ValueError("embedded system template differs from the training prompt")
        return template
