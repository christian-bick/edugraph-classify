"""Small W&B boundary: scalar metrics and references, never tensor uploads."""
from __future__ import annotations

def scalar_metrics(values: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in values.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name + "/"))
        elif type(value) in (int, float):
            result[name] = value
    return result


class WandbTracker:
    def __init__(self, manifest: dict, *, resume=False, sdk=None):
        if sdk is None:
            import wandb as sdk
        self.sdk = sdk
        config = manifest["recipe"]
        tracking = config["tracking"]
        # Explicit configuration only. No environment, source capture or model.watch.
        public = {key: value for key, value in manifest.items() if key != "files_sha256"}
        self.run = sdk.init(entity=tracking["entity"], project=tracking["project"],
            id=config["run_id"], name=config["run_id"], config=public,
            resume="allow" if resume else "never", mode="online",
            settings=sdk.Settings(disable_code=True, disable_git=True, console="off"))
        self.run.define_metric("optimizer_step")
        self.run.define_metric("*", step_metric="optimizer_step")

    def log(self, values: dict, step: int):
        self.run.log({**scalar_metrics(values), "optimizer_step": step})

    def checkpoint(self, reference: dict, *, final=False):
        artifact = self.sdk.Artifact(self.run.id + ("-model" if final else "-checkpoint"),
                                    type="model" if final else "checkpoint", metadata=reference)
        # SHA256 is our integrity contract; checksum=False avoids another GCS read
        # and prevents W&B from needing Google credentials itself.
        artifact.add_reference(reference["uri"], name="bundle.tar", checksum=False)
        self.run.log_artifact(artifact)

    def finish(self, success: bool):
        self.run.finish(exit_code=0 if success else 1)
