"""Read-only retrieval and independent verification of completed RunPod runs."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import tarfile
from collections import Counter
from pathlib import Path

from .checkpoint_storage import unpack_verified
from .dataset import file_sha256
from .evaluation import evaluate
from .learning_data import safe_relative, verify_prepared, write_json
from .ontology import OntologyCatalog
from .self_hosted_run import resume_identity, verify_resume


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_completion(manifest: dict, completion: dict) -> dict:
    """Bind a completion marker and both immutable references to this run."""
    if (not isinstance(completion, dict) or completion.get("status") != "completed"
            or completion.get("run_id") != manifest["run_id"]
            or completion.get("identity") != resume_identity(manifest)):
        raise ValueError("completion does not match the prepared run identity")
    if (any(type(completion.get(key)) is not int or completion[key] <= 0
            for key in ("selected_epoch", "epochs_completed", "step"))
            or not completion["selected_epoch"] <= completion["epochs_completed"] <= manifest["recipe"]["training"]["epochs"]):
        raise ValueError("completion has invalid training counters")
    root = manifest["recipe"]["artifacts"]["root_uri"].rstrip("/")
    for key, prefix in (("model", "models"), ("continuation", "checkpoints")):
        ref = completion.get(key)
        if (not isinstance(ref, dict) or ref.get("run_id") != manifest["run_id"]
                or not isinstance(ref.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])
                or type(ref.get("bytes")) is not int or ref["bytes"] <= 0
                or ref.get("uri") != f"{root}/{prefix}/{manifest['run_id']}/{ref['sha256']}.tar"):
            raise ValueError(f"invalid {key} bundle reference for this run")
    return completion


def _verify_extraction(archive: Path, directory: Path) -> None:
    """Recheck cached extracted files; a checksum of the archive alone is insufficient."""
    if directory.is_symlink():
        raise ValueError("extracted bundle must not be a link")
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        names = [safe_relative(member.name) for member in members]
        if len(names) != len(set(names)) or any(not member.isfile() for member in members):
            raise ValueError("bundle must contain unique regular files only")
        actual = set()
        for path in directory.rglob("*"):
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("extracted bundle contains links or escaped paths")
            if path.is_file():
                actual.add(path.relative_to(directory).as_posix())
        if actual != set(names):
            raise ValueError("extracted bundle file inventory differs from its archive")
        for member in members:
            with bundle.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if file_sha256(directory / member.name) != digest:
                raise ValueError("extracted bundle content differs from its archive")


def retrieve_result(manifest_path: Path, output: Path, store, *, completion: dict | None = None) -> dict:
    """Download completed bundles without uploads, provider calls, or tensor loading."""
    manifest = _read(manifest_path)
    if completion is None:
        uri = manifest["recipe"]["artifacts"]["root_uri"].rstrip("/")
        completion = store.read_json(f"{uri}/runs/{manifest['run_id']}/completed.json")
    validate_completion(manifest, completion)
    output.mkdir(parents=True, exist_ok=True)
    bundles = {}
    for kind, key in (("model", "model"), ("checkpoint", "continuation")):
        ref = completion[key]
        archive = output / f"{kind}-{ref['sha256']}.tar"
        directory = archive.with_suffix("")
        if archive.is_symlink():
            raise ValueError("cached bundle archive must not be a link")
        if not archive.exists():
            partial = archive.with_suffix(".partial")
            if partial.exists() or partial.is_symlink():
                raise ValueError("unfinished download exists; inspect it before retrying")
            try:
                store.download(ref, partial)
                if file_sha256(partial) != ref["sha256"] or partial.stat().st_size != ref["bytes"]:
                    raise ValueError("downloaded bundle checksum or size mismatch")
                partial.replace(archive)
            finally:
                partial.unlink(missing_ok=True)
        if file_sha256(archive) != ref["sha256"] or archive.stat().st_size != ref["bytes"]:
            raise ValueError("cached bundle checksum or size mismatch")
        if not directory.exists():
            unpack_verified(archive, ref["sha256"], directory)
        _verify_extraction(archive, directory)
        bundles[kind] = {"reference": ref, "archive": str(archive.resolve()),
                         "directory": str(directory.resolve()), "sha256_verified": True}
    result = {"run_id": manifest["run_id"], "completion": completion, "bundles": bundles,
              "model_dir": bundles["model"]["directory"], "checkpoint_dir": bundles["checkpoint"]["directory"]}
    write_json(output / "completion.json", completion)
    write_json(output / "retrieval.json", result)
    return result


def verify_retrieved(manifest_path: Path, output: Path) -> dict:
    """Revalidate local archives and their contents before offline result inspection."""
    manifest = _read(manifest_path)
    completion = validate_completion(manifest, _read(output / "completion.json"))
    directories = {}
    for kind, key in (("model", "model"), ("checkpoint", "continuation")):
        ref = completion[key]
        archive = output / f"{kind}-{ref['sha256']}.tar"
        directory = archive.with_suffix("")
        if archive.is_symlink() or file_sha256(archive) != ref["sha256"] or archive.stat().st_size != ref["bytes"]:
            raise ValueError("cached bundle checksum or size mismatch")
        _verify_extraction(archive, directory)
        directories[kind] = directory
    return verify_result(manifest_path, directories["model"], directories["checkpoint"], completion)


def _adapter(path: Path, *, selected: Path | None = None) -> dict:
    import torch
    from safetensors import safe_open

    shapes, parameters = {}, 0
    with safe_open(path, framework="pt", device="cpu") as weights:
        names = list(weights.keys())
        if not names or any(not re.fullmatch(
                r"base_model\.model\.model\.language_model\..+\.lora_[AB]\.weight", name) for name in names):
            raise ValueError("adapter contains non-language parameters or is empty")
        for name in names:
            tensor = weights.get_tensor(name)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("adapter contains nonfinite tensors")
            shapes[name] = tuple(tensor.shape)
            parameters += tensor.numel()
        if selected is not None:
            with safe_open(selected, framework="pt", device="cpu") as best:
                if set(best.keys()) != set(names) or any(
                        not torch.equal(weights.get_tensor(name), best.get_tensor(name)) for name in names):
                    raise ValueError("exported adapter differs from the selected checkpoint adapter")
    return {"shapes": shapes, "parameters": parameters, "tensors": len(names)}


def _verify_optimizer(path: Path, state: dict, adapter: dict, gpu_count: int) -> int:
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    optimizer = saved["optimizer"]
    states, groups = optimizer["state"], optimizer["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != adapter["tensors"]:
        raise ValueError("optimizer parameter groups differ from adapter tensors")
    if set(states) != set(groups[0]["params"]) or len(states) != adapter["tensors"]:
        raise ValueError("optimizer states do not cover every trainable tensor")
    shapes = Counter()
    for item in states.values():
        if (not torch.is_tensor(item["step"]) or item["step"].numel() != 1
                or item["step"].item() != state["step"]
                or item["exp_avg"].shape != item["exp_avg_sq"].shape
                or not bool(torch.isfinite(item["exp_avg"]).all())
                or not bool(torch.isfinite(item["exp_avg_sq"]).all())):
            raise ValueError("optimizer moments or step are invalid")
        shapes[tuple(item["exp_avg"].shape)] += 1
    if shapes != Counter(adapter["shapes"].values()):
        raise ValueError("optimizer moment shapes differ from adapter tensors")
    cpu_rng, cuda_rng = saved["torch_rng"], saved["cuda_rng"]
    if (not torch.is_tensor(cpu_rng) or cpu_rng.dtype != torch.uint8 or cpu_rng.ndim != 1
            or not isinstance(cuda_rng, list) or len(cuda_rng) != gpu_count
            or any(not torch.is_tensor(rng) or rng.dtype != torch.uint8 or rng.ndim != 1
                   or rng.numel() == 0 for rng in cuda_rng)):
        raise ValueError("checkpoint lacks valid CPU/CUDA RNG state")
    # Validate without mutating this process's global random streams.
    torch.Generator().set_state(cpu_rng)
    random.Random().setstate(saved["python_rng"])
    return len(states)


def _reproduce_metrics(model: Path, examples: list[dict], epochs: int, catalog) -> dict:
    train = [row for row in examples if row["split"] == "train"]
    cohorts = {name: [row for row in examples if row["split"] == name]
               for name in ("validation", "final_validation")}
    cohorts["train_diagnostic"] = [row for row in train if row["train_diagnostic"]]
    phases = {"baseline": ("validation", "train_diagnostic"),
              **{f"epoch-{epoch}": ("validation", "train_diagnostic") for epoch in range(1, epochs + 1)}}
    if cohorts["final_validation"]:
        phases["final"] = ("final_validation",)
    reports = model / "reports"
    for suffix in ("metrics", "predictions"):
        observed = {path.name.removesuffix(f"-{suffix}.json") for path in reports.glob(f"*-{suffix}.json")}
        if observed != set(phases):
            raise ValueError("evaluation report phases differ from completed training")
    metrics = {}
    for phase, names in phases.items():
        predictions = _read(reports / f"{phase}-predictions.json")
        recorded = _read(reports / f"{phase}-metrics.json")
        expected = {row["id"] for name in names for row in cohorts[name]}
        actual = [row["id"] for row in predictions]
        if set(recorded) != set(names) or len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("evaluation report cohorts or prediction IDs differ from prepared inputs")
        metrics[phase] = {}
        for name in names:
            ids = {row["id"] for row in cohorts[name]}
            measured = evaluate(cohorts[name], [row for row in predictions if row["id"] in ids], train, catalog)
            if measured != recorded[name]:
                raise ValueError("exported metrics differ from independent reproduction")
            metrics[phase][name] = measured
    return metrics


def verify_result(manifest_path: Path, model_dir: Path, checkpoint_dir: Path, completion: dict) -> dict:
    """Inspect already authenticated bundles; use verify_retrieved to verify archives too."""
    original = _read(manifest_path)
    manifest, examples = verify_prepared(manifest_path, original["code_commit"])
    validate_completion(manifest, completion)
    if any(_read(root / "manifest.json") != manifest for root in (model_dir, checkpoint_dir)):
        raise ValueError("result bundle manifest differs from prepared inputs")
    portable = {"prompt.json", "schema.json", "closed_schema.json", "chat_template.jinja"}
    for name, digest in manifest["files_sha256"].items():
        if name in portable or name.startswith("processor/"):
            if file_sha256(model_dir / safe_relative(name)) != digest:
                raise ValueError("exported processor, prompt or schema differs from prepared inputs")
    if _read(model_dir / "result.json") != {key: value for key, value in completion.items() if key != "model"}:
        raise ValueError("exported result differs from its completion marker")
    state = verify_resume(manifest, checkpoint_dir)
    epochs, step = completion.get("epochs_completed"), completion.get("step")
    selected = completion.get("selected_epoch")
    train_count = sum(row["split"] == "train" for row in examples)
    if (type(epochs) is not int or not 1 <= epochs <= manifest["recipe"]["training"]["epochs"]
            or type(selected) is not int or not 1 <= selected <= epochs
            or type(step) is not int or step != epochs * math.ceil(train_count / manifest["recipe"]["training"]["batch_size"])
            or state["run_id"] != manifest["run_id"] or state["phase"] != "final" or state["offset"] != 0
            or state["epoch"] != epochs or state["step"] != step or state["best"]["epoch"] != selected):
        raise ValueError("completion and checkpoint training positions disagree")
    weights_name = "adapter_model.safetensors"
    exported = _adapter(model_dir / "adapter" / weights_name, selected=checkpoint_dir / "best-adapter" / weights_name)
    current = _adapter(checkpoint_dir / "adapter" / weights_name)
    if current != exported:
        raise ValueError("current and selected adapter tensor inventories differ")
    if exported["tensors"] != 2 * manifest["language_target_audit"]["language_linear_modules"]:
        raise ValueError("adapter tensor count differs from the prepared language target audit")
    audit = _read(model_dir / "reports/runtime-audit.json")
    checkpoint_audit = _read(checkpoint_dir / "trainable-audit.json")
    expected_names = set(exported["shapes"])
    for observed in (audit, checkpoint_audit):
        names = {re.sub(r"(\.lora_[AB])\.default\.", r"\1.", name) for name in observed["parameter_names"]}
        if (observed["vision_or_bridge_trainable_parameters"] != 0
                or observed["trainable_parameters"] != exported["parameters"]
                or observed["trainable_tensors"] != exported["tensors"] or names != expected_names):
            raise ValueError("runtime language-only parameter audit differs from adapter tensors")
    optimizer_states = _verify_optimizer(checkpoint_dir / "optimizer.pt", state, current,
                                         manifest["recipe"]["execution"]["gpu_count"])
    catalog = OntologyCatalog.load(manifest["recipe"]["ontology_version"])
    if catalog.snapshot_sha256 != manifest["ontology_snapshot_sha256"]:
        raise ValueError("ontology snapshot differs from the prepared run")
    metrics = _reproduce_metrics(model_dir, examples, epochs, catalog)
    for phase in metrics:
        if phase != "final":
            for suffix in ("metrics", "predictions"):
                name = f"reports/{phase}-{suffix}.json"
                if file_sha256(checkpoint_dir / name) != file_sha256(model_dir / name):
                    raise ValueError("continuation evaluation history differs from the selected export")
    scores = {epoch: [metrics[f"epoch-{epoch}"]["validation"]["explicit"][name]
                     for name in ("exact_set_match", "f1")] for epoch in range(1, epochs + 1)}
    best_epoch = max(scores, key=scores.get)
    baseline = metrics["baseline"]["validation"]["explicit"]
    best = metrics[f"epoch-{selected}"]["validation"]["explicit"]
    if (selected != best_epoch or state["best"]["score"] != scores[selected]
            or completion.get("baseline_validation") != baseline or completion.get("selected_validation") != best
            or completion.get("validation_delta") != {key: best[key] - baseline[key] for key in ("exact_set_match", "f1")}):
        raise ValueError("selected checkpoint or completion metrics disagree with reproduced evaluation")
    return {"run_id": manifest["run_id"], "status": "verified", "code_commit": manifest["code_commit"],
            "selected_epoch": selected, "epochs_completed": epochs, "optimizer_steps": step,
            "model": completion["model"], "continuation": completion["continuation"],
            "adapter_tensors": exported["tensors"], "trainable_parameters": exported["parameters"],
            "optimizer_states": optimizer_states, "vision_or_bridge_trainable_parameters": 0,
            "all_metrics_reproduced": True, "continuation_verified": True,
            "metrics": metrics, "gpu_resume_executed": False, "termination_verified": False,
            "quality_promotion": False}
