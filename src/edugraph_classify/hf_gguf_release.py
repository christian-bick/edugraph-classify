"""Prepare a hash-verified, curated Hugging Face GGUF release offline."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path


PUBLIC_FILES = ("model-Q4_K_M.gguf", "mmproj-BF16.gguf", "chat_template.jinja",
                "prompt.json", "closed_schema.json")
SOURCE_FILES = (*PUBLIC_FILES, "manifest.json")
ARCHIVE_FILES = {*SOURCE_FILES, "candidate.json"}
SMALL_FILES = {"candidate.json", "chat_template.jinja", "prompt.json",
               "closed_schema.json", "manifest.json"}
MAX_SMALL_FILE_BYTES = 1_000_000
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
LICENSE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,63}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _keys(value: object, keys: set[str], section: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid {section} fields")


def _digest(value: object, section: str, pattern=SHA256) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"invalid {section} pin")


def load_release_recipe(path: Path) -> dict:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    _keys(recipe, {"kind", "repository", "candidate", "base_model", "dataset",
                   "ontology_version", "llama_cpp_commit", "benchmark_code_commit", "benchmark_record_commit",
                   "files_sha256", "evaluation", "model_license"}, "release recipe")
    if recipe["kind"] != "huggingface_gguf_release_v1" or not REPO.fullmatch(recipe["repository"]):
        raise ValueError("invalid release repository or recipe kind")
    candidate = recipe["candidate"]
    _keys(candidate, {"run_id", "archive_sha256", "archive_bytes", "source_model_sha256",
                      "source_training_run_id"}, "candidate")
    _digest(candidate["archive_sha256"], "candidate archive")
    _digest(candidate["source_model_sha256"], "source model")
    if type(candidate["archive_bytes"]) is not int or candidate["archive_bytes"] <= 0:
        raise ValueError("invalid candidate archive size")
    for field in ("run_id", "source_training_run_id"):
        if not isinstance(candidate[field], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,127}", candidate[field]):
            raise ValueError(f"invalid candidate {field}")
    _keys(recipe["base_model"], {"repository", "revision"}, "base model")
    if not REPO.fullmatch(recipe["base_model"]["repository"]):
        raise ValueError("invalid base model repository")
    _digest(recipe["base_model"]["revision"], "base model revision", COMMIT)
    _keys(recipe["model_license"], {"id", "upstream_license_sha256"}, "model license")
    if (not isinstance(recipe["model_license"]["id"], str) or
            not LICENSE_ID.fullmatch(recipe["model_license"]["id"])):
        raise ValueError("invalid model license ID pin")
    _digest(recipe["model_license"]["upstream_license_sha256"], "upstream model license")
    _keys(recipe["dataset"], {"repository", "release", "revision"}, "dataset")
    if not REPO.fullmatch(recipe["dataset"]["repository"]):
        raise ValueError("invalid dataset repository")
    _digest(recipe["dataset"]["revision"], "dataset revision", COMMIT)
    for field in ("llama_cpp_commit", "benchmark_code_commit", "benchmark_record_commit"):
        _digest(recipe[field], field, COMMIT)
    if not isinstance(recipe["ontology_version"], str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", recipe["ontology_version"]):
        raise ValueError("invalid ontology version")
    _keys(recipe["files_sha256"], set(SOURCE_FILES), "candidate file hashes")
    for name, digest in recipe["files_sha256"].items():
        _digest(digest, name)
    evaluation = recipe["evaluation"]
    _keys(evaluation, {"validation_count", "nf4_exact", "bf16_exact", "gguf_exact",
                       "gguf_micro_f1_percent", "gguf_invalid", "gguf_report_sha256",
                       "nf4_final_exact", "nf4_final_count"}, "evaluation")
    if (type(evaluation["validation_count"]) is not int or evaluation["validation_count"] <= 0 or
            any(type(evaluation[field]) is not int or not 0 <= evaluation[field] <= evaluation["validation_count"]
                for field in ("nf4_exact", "bf16_exact", "gguf_exact", "gguf_invalid")) or
            type(evaluation["nf4_final_count"]) is not int or evaluation["nf4_final_count"] <= 0 or
            type(evaluation["nf4_final_exact"]) is not int or not 0 <= evaluation["nf4_final_exact"] <= evaluation["nf4_final_count"] or
            type(evaluation["gguf_micro_f1_percent"]) not in (int, float) or
            not 0 <= evaluation["gguf_micro_f1_percent"] <= 100):
        raise ValueError("invalid evaluation values")
    _digest(evaluation["gguf_report_sha256"], "GGUF report")
    return recipe


def _card(recipe: dict, license_id: str | None) -> str:
    candidate, evaluation = recipe["candidate"], recipe["evaluation"]
    license_metadata = f"license: {license_id}\n" if license_id else ""
    draft = "\n> **DRAFT — DO NOT PUBLISH.** The model-weight license has not been selected.\n" if not license_id else ""
    license_section = (f"""\n## License and attribution

The EduGraph fine-tuned and Q4_K_M-converted model weights are released under Apache License 2.0. The included `LICENSE` is the exact license text from the [pinned upstream checkpoint](https://huggingface.co/{recipe['base_model']['repository']}/blob/{recipe['base_model']['revision']}/LICENSE) (SHA-256 `{recipe['model_license']['upstream_license_sha256']}`), including the Alibaba Cloud copyright notice.

**Modifications:** EduGraph trained a language adapter, merged it into the base weights, converted the result to GGUF, and quantized the language model. The vision projector remains BF16. The dataset and ontology have separate upstream ownership and terms.
""" if license_id else "")
    return f"""---
{license_metadata}pipeline_tag: image-text-to-text
library_name: gguf
base_model: {recipe['base_model']['repository']}
datasets:
  - {recipe['dataset']['repository']}
tags:
  - gguf
  - llama-cpp
  - edugraph
---

# EduGraph Classifier — Qwen3.8-27B GGUF
{draft}
This research model labels **one educational task image** with explicit EduGraph ontology identifiers in `areas`, `scopes`, and `abilities`. It is a language-adapter fine-tune of [{recipe['base_model']['repository']}](https://huggingface.co/{recipe['base_model']['repository']}) at revision `{recipe['base_model']['revision']}`, merged into the BF16 base and converted to Q4_K_M GGUF. The vision projector remains BF16. The two GGUF files in this repository are a pair; use both.

## Files and provenance

- Language model: `model-Q4_K_M.gguf` (SHA-256 `{recipe['files_sha256']['model-Q4_K_M.gguf']}`)
- Vision projector: `mmproj-BF16.gguf` (SHA-256 `{recipe['files_sha256']['mmproj-BF16.gguf']}`)
- Pinned llama.cpp converter/runtime commit: `{recipe['llama_cpp_commit']}`
- Selected training run: `{candidate['source_training_run_id']}`; GGUF validation run: `{candidate['run_id']}` at code commit `{recipe['benchmark_code_commit']}`
- Training dataset: [{recipe['dataset']['repository']}](https://huggingface.co/datasets/{recipe['dataset']['repository']}) release `{recipe['dataset']['release']}` at revision `{recipe['dataset']['revision']}`; ontology `{recipe['ontology_version']}`
- `provenance.json` contains the source archive and curated-file checksums. The [immutable benchmark record](https://github.com/christian-bick/edugraph-classify/blob/{recipe['benchmark_record_commit']}/docs/RUNPOD-GGUF-BENCHMARK-20261004.md) documents the corrected v4 score; the execution code remains pinned separately above.

## Use with llama.cpp

Use the included `chat_template.jinja`, `prompt.json`, and `closed_schema.json` together. The template injects the fixed classifier system prompt and disables thinking. The measured single-request path used a 2,048-token context, 1,024-token microbatch, GPU offload, greedy generation, a 512-token output ceiling, and an RGB JPEG image encoded at quality 75. Start a local server:

```bash
llama-server -m model-Q4_K_M.gguf --mmproj mmproj-BF16.gguf \\
  --jinja --chat-template-file chat_template.jinja \\
  --ctx-size 2048 --ubatch-size 1024 --parallel 1 \\
  --n-gpu-layers all --fit off --host 127.0.0.1 --port 8080
```

Send one user message containing the text from `prompt.json`'s `user` field and the image as a JPEG data URL to `/v1/chat/completions`. Set `temperature: 0`, `max_tokens: 512`, `chat_template_kwargs: {{"enable_thinking": false}}`, and `response_format: {{"type": "json_object", "schema": ...}}`. Before sending the schema, construct a copy with `properties` in **`areas`, `scopes`, `abilities` order**. The saved schema may be serialized in a different key order; this request-time ordering is essential to reproduce the measured result. See [`completion_request`](https://github.com/christian-bick/edugraph-classify/blob/{recipe['benchmark_code_commit']}/src/edugraph_classify/inference_benchmark.py) for the exact request builder.

The model returns explicit labels only. Do not treat ontology `integrates` relations or derived ancestors as model predictions. Verify labels against the pinned ontology version before downstream use.
{license_section}

## Evaluation

All validation rows below are the **same {evaluation['validation_count']} original images** and gold labels; exact-set match compares complete explicit label sets. This is a diagnostic cohort used for checkpoint selection, not a fresh blind test.

| Configuration | Exact-set match |
|---|---:|
| Selected NF4 base plus adapter (training runtime) | {evaluation['nf4_exact']}/{evaluation['validation_count']} |
| Merged BF16 (Transformers runtime) | {evaluation['bf16_exact']}/{evaluation['validation_count']} |
| This Q4_K_M GGUF (llama.cpp, corrected property order) | **{evaluation['gguf_exact']}/{evaluation['validation_count']}** |

The GGUF run measured {evaluation['gguf_micro_f1_percent']:.2f}% micro label F1 and {evaluation['gguf_invalid']}/{evaluation['validation_count']} invalid outputs. Its immutable report archive has SHA-256 `{evaluation['gguf_report_sha256']}`. The separate **{evaluation['nf4_final_exact']}/{evaluation['nf4_final_count']} final-assessment result belongs only to the NF4-plus-adapter configuration**; this GGUF export has not been scored on that final cohort. Differences between BF16 and GGUF also include conversion, serving, and JSON constraint mechanics, so they do not isolate quantization loss.

## Limitations

Training and evaluation used synthetic educational-task images and ontology `{recipe['ontology_version']}`. Real-world classroom performance, unfamiliar label combinations, and a fresh post-refinement assessment are unmeasured. The GGUF run fit an L40S; its memory and timing do not establish fit, load behavior, or cold-start performance on a 24 GB GCP L4. This model is a research candidate and should be reviewed for the intended educational setting before use in decisions affecting learners.
"""


def prepare_release(archive_path: Path, recipe: dict, output: Path, *,
                    license_id: str | None = None, license_file: Path | None = None) -> dict:
    """Verify the immutable source, then atomically stage only reviewed public files."""
    if (license_id is None) != (license_file is None):
        raise ValueError("model license ID and license file must be supplied together")
    if license_id is not None and not LICENSE_ID.fullmatch(license_id):
        raise ValueError("invalid model license ID")
    if license_file is not None and (not license_file.is_file() or license_file.stat().st_size == 0):
        raise ValueError("model license file is missing or empty")
    if license_id is not None and license_id != recipe["model_license"]["id"]:
        raise ValueError("model license ID differs from release recipe")
    if license_file is not None and _sha256(license_file) != recipe["model_license"]["upstream_license_sha256"]:
        raise ValueError("model license file differs from pinned upstream LICENSE")
    candidate = recipe["candidate"]
    if archive_path.stat().st_size != candidate["archive_bytes"] or _sha256(archive_path) != candidate["archive_sha256"]:
        raise ValueError("candidate archive differs from its size or SHA-256 pin")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"release directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        if (len(members) != len(ARCHIVE_FILES) or
                {member.name for member in members} != ARCHIVE_FILES or
                any(not member.isfile() for member in members)):
            raise ValueError("candidate archive has unexpected, duplicate, or unsafe members")
        info = {member.name: member for member in members}
        if any(info[name].size > MAX_SMALL_FILE_BYTES for name in SMALL_FILES):
            raise ValueError("candidate metadata or contract file is unexpectedly large")
        with archive.extractfile(info["candidate.json"]) as source:
            metadata = json.load(source)
        if (not isinstance(metadata, dict) or
                not isinstance(metadata.get("source_model"), dict) or
                not isinstance(metadata.get("base_model"), dict) or
                metadata.get("kind") != "gguf_inference_candidate_v1" or
                metadata.get("run_id") != candidate["run_id"] or
                metadata.get("source_training_run_id") != candidate["source_training_run_id"] or
                metadata["source_model"].get("sha256") != candidate["source_model_sha256"] or
                metadata["base_model"].get("hf_repository") != recipe["base_model"]["repository"] or
                metadata["base_model"].get("hf_revision") != recipe["base_model"]["revision"] or
                metadata.get("quantization") != "Q4_K_M" or
                metadata.get("llama_cpp_commit") != recipe["llama_cpp_commit"] or
                metadata.get("files_sha256") != recipe["files_sha256"]):
            raise ValueError("candidate metadata differs from the release recipe")
        temporary = Path(tempfile.mkdtemp(prefix=".hf-gguf-release-", dir=output.parent))
        try:
            for name in SOURCE_FILES:
                digest = hashlib.sha256()
                with archive.extractfile(info[name]) as source:
                    if name in PUBLIC_FILES:
                        with (temporary / name).open("xb") as target:
                            for block in iter(lambda: source.read(1024 * 1024), b""):
                                digest.update(block)
                                target.write(block)
                    else:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block)
                if digest.hexdigest() != recipe["files_sha256"][name]:
                    raise ValueError(f"candidate member checksum mismatch: {name}")
            if license_file is not None:
                shutil.copyfile(license_file, temporary / "LICENSE")
            (temporary / "README.md").write_text(_card(recipe, license_id), encoding="utf-8", newline="\n")
            provenance = {"kind": "edugraph_hf_gguf_provenance_v1", "repository": recipe["repository"],
                          "candidate_archive_sha256": candidate["archive_sha256"],
                          "candidate_archive_bytes": candidate["archive_bytes"],
                          "candidate_run_id": candidate["run_id"],
                          "source_training_run_id": candidate["source_training_run_id"],
                          "source_model_sha256": candidate["source_model_sha256"],
                          "base_model": recipe["base_model"], "dataset": recipe["dataset"],
                          "ontology_version": recipe["ontology_version"],
                          "llama_cpp_commit": recipe["llama_cpp_commit"],
                          "benchmark_code_commit": recipe["benchmark_code_commit"],
                          "benchmark_record_commit": recipe["benchmark_record_commit"],
                          "source_files_sha256": recipe["files_sha256"],
                          "gguf_report_sha256": recipe["evaluation"]["gguf_report_sha256"],
                          "model_license": license_id,
                          "model_license_sha256": recipe["model_license"]["upstream_license_sha256"] if license_id else None,
                          "license_gate": "passed" if license_id else "blocked_owner_choice"}
            (temporary / "provenance.json").write_text(
                json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
            if license_id is None:
                (temporary / "DO_NOT_PUBLISH_LICENSE_PENDING.txt").write_text(
                    "The model-weight license is an owner decision. Choose its Hub license ID and supply the corresponding license file before publication.\n",
                    encoding="utf-8", newline="\n")
            temporary.rename(output)
        except BaseException:
            shutil.rmtree(temporary)
            raise
    return {"repository": recipe["repository"], "output": str(output),
            "license_gate": provenance["license_gate"],
            "files_sha256": {name: _sha256(output / name) for name in PUBLIC_FILES}}


def add_hf_gguf_release_parser(commands) -> None:
    parser = commands.add_parser("hf-gguf-release", help="prepare a verified Hugging Face GGUF release locally")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--candidate-tar", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-license-id")
    parser.add_argument("--model-license-file")


def run_hf_gguf_release_command(args) -> tuple[int, dict]:
    output = Path(args.output).resolve()
    root = Path.cwd().resolve()
    if output.is_relative_to(root) and (not output.relative_to(root).parts or
                                        output.relative_to(root).parts[0] not in {"artifacts", "runs", "reports", "temp"}):
        raise ValueError("release output inside the repository must use an ignored artifact directory")
    recipe = load_release_recipe(Path(args.recipe))
    return 0, prepare_release(Path(args.candidate_tar), recipe, output,
                              license_id=args.model_license_id,
                              license_file=Path(args.model_license_file) if args.model_license_file else None)
