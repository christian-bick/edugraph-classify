"""Thin adapter for the Fireworks pooled training API (never provisions GPUs)."""

from __future__ import annotations

import asyncio
import base64
import time

import httpx

from ..qwen_rendering import VisionExample

API_ROOT = "https://api.fireworks.ai"


def inspect_training_model(model: str, api_key: str, *, client=None) -> dict:
    def inspect(http):
        response = http.get(f"{API_ROOT}/v1/{model}", headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        source = response.json()
        fields = ("name", "state", "supportsImageInput", "supportsLora", "useTrainingV2",
                  "supervisedLoraTunable", "trainingContextLength", "updateTime")
        result = {field: source.get(field) for field in fields}
        if result["name"] != model or result["state"] != "READY" or any(
            result[field] is not True for field in ("supportsImageInput", "supportsLora", "useTrainingV2", "supervisedLoraTunable")
        ):
            raise ValueError("live catalog does not explicitly support this vision LoRA model on training V2")
        return result
    if client is not None:
        return inspect(client)
    with httpx.Client(timeout=30) as http:
        return inspect(http)


def to_datum(example: VisionExample, types):
    targets, weights = example.targets_and_weights()
    chunks = [types.EncodedTextChunk(tokens=list(example.before_image)),
              types.ImageChunk(data=example.image, format="jpeg", expected_tokens=example.image_tokens),
              types.EncodedTextChunk(tokens=[*example.after_image, *example.completion[:-1]])]
    return types.Datum(
        model_input=types.ModelInput(chunks=chunks),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=targets, dtype="int64", shape=[len(targets)]),
            "weights": types.TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )


class FireworksTrainingAPI:
    def __init__(self, api_key: str, model: str, tokenizer, *, service_factory=None, platform_factory=None, types=None):
        if service_factory is None:
            from fireworks.training.sdk import FiretitanServiceClient
            from fireworks.training.sdk.fireworks_client import FireworksClient
            from tinker import types as tinker_types
            service_factory, platform_factory, types = FiretitanServiceClient, FireworksClient, tinker_types
        self.key, self.model, self.tokenizer = api_key, model, tokenizer
        self.service_factory, self.platform_factory, self.types = service_factory, platform_factory, types
        self.service = self.training = self.sampler = None
        self.expected_target_modules = None

    def start(self, config: dict, *, resume_state: str | None = None) -> dict:
        self.expected_target_modules = config.get("expected_target_modules")
        self.service = self.service_factory(api_key=self.key, base_url=f"{API_ROOT}/training/v1/serverless", timeout=60, max_retries=0)
        if resume_state is None:
            self.training = self.service.create_lora_training_client(
                base_model=self.model, rank=config["rank"], alpha=config["alpha"], seed=config["seed"],
                train_attn=config.get("train_attn", True), train_mlp=config.get("train_mlp", True),
                train_unembed=config.get("train_unembed", True),
            )
        else:
            self.training = self.service.create_training_client_from_state_with_optimizer(resume_state)
        return {"training_session_id": self.service.training_session_id,
                "training_session_name": self.service.training_session_name,
                "run_id": self.training.run_id,
                "resumed_from_training_state": resume_state,
                "lora_policy": {name: config.get(name, True) for name in ("train_attn", "train_mlp", "train_unembed")}}

    def train_batch(self, examples: list[VisionExample], config: dict) -> dict:
        result = self.training.forward_backward([to_datum(x, self.types) for x in examples], "cross_entropy").result(timeout=180)
        metrics = {key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))}
        # Fail before stepping if the provider reports a non-finite objective.
        import math
        if not metrics or not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("training returned missing or non-finite metrics")
        self.training.optim_step(self.types.AdamParams(**{key: config[key] for key in
            ("learning_rate", "beta1", "beta2", "eps", "weight_decay")})).result(timeout=180)
        return metrics

    def checkpoint(self, name: str, *, sampler: bool = False) -> str:
        save = self.training.save_weights_for_sampler if sampler else self.training.save_state
        return save(name).result(timeout=180).path

    def use_sampler(self, checkpoint: str | None = None) -> None:
        if self.sampler is not None:
            self.sampler.close()
        options = {"model_path": checkpoint} if checkpoint else {"base_model": self.model}
        self.sampler = self.service.create_sampling_client(**options, tokenizer=self.tokenizer)

    def predict(self, example: VisionExample, config: dict) -> dict:
        # SDK sample() calls ModelInput.to_ints(), which rejects ImageChunk.
        # Its completions transport explicitly supports token prompts + images.
        output = ({"response_format": {"type": "json_schema", "json_schema": {
            "name": "edugraph_labels", "schema": config["output_schema"],
        }}} if "output_schema" in config else {})
        async def sample():
            return await asyncio.wait_for(self.sampler.deployment_sampler.async_completions_stream(
                prompt=example.sampling_tokens(),
                images=["data:image/jpeg;base64," + base64.b64encode(example.image).decode("ascii")],
                max_tokens=config["max_tokens"], temperature=config["temperature"], seed=config["seed"],
                stop=["<|im_end|>"], raw_output=True, return_token_ids=True, http_timeout=120, hotload_max_retries=0,
                **output,
            ), timeout=150)
        start = time.monotonic()
        raw, _ = asyncio.run(sample())
        return {"text": raw["choices"][0]["text"], "finish_reason": raw["choices"][0].get("finish_reason"),
                "usage": raw.get("usage", {}), "latency_seconds": time.monotonic() - start, "raw": raw}

    def retain_checkpoint(self, path: str, output_model_id: str) -> dict:
        """Register the final private adapter before pooled session expiry.

        Fireworks calls this promotion. It preserves an experimental artifact;
        it is not a quality approval, publication, or inference deployment.
        """
        with self.platform_factory(api_key=self.key, base_url=API_ROOT) as platform:
            rows = platform.list_training_session_checkpoints(self.service.training_session_name)
            # Backend path is account/run-id/logical-name-session-suffix;
            # control-plane IDs flatten the last two segments with a hyphen.
            checkpoint_id = "-".join(path.rsplit("/", 2)[-2:])
            matches = [row for row in rows if row.get("promotable") is True and row["name"].rsplit("/", 1)[-1] == checkpoint_id]
            if len(matches) != 1:
                raise ValueError("final sampler checkpoint could not be resolved for durable retention")
            model = platform.promote_session_checkpoint(matches[0]["name"], output_model_id, self.model)
            if self.expected_target_modules is not None and sorted(model.get("peftDetails", {}).get("targetModules", [])) != sorted(self.expected_target_modules):
                raise ValueError("retained adapter target modules differ from the language-only policy")
            return {key: model.get(key) for key in ("name", "state", "kind", "public", "peftDetails")}

    def close(self) -> None:
        try:
            if self.sampler is not None:
                self.sampler.close()
        finally:
            if self.service is not None:
                try:
                    self.service.close()
                finally:
                    self.service.holder.close()
