from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from edugraph_classify import cli, training
from edugraph_classify.contracts import ModelIdentity
from edugraph_classify.preflight import EligibilityStatus
from edugraph_classify.providers.fireworks import ModelResolutionError, TrainingLaunchError
from edugraph_classify.providers.vertex import VertexLaunchError
from edugraph_classify.vertex_artifacts import VertexArtifactError
from edugraph_classify.training import (
    DatasetUploadSpec,
    ProviderDatasetRecord,
    SupervisedFineTuningSpec,
    TrainingConfigError,
    TrainingJobRecord,
)


def test_fireworks_preflight_loads_env_and_returns_fail_closed_status(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FIREWORKS_API_KEY=test-only\n", encoding="utf-8")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    model = SimpleNamespace(
        name="accounts/fireworks/models/qwen3p5-9b",
        display_name="Qwen3.5 9B",
        supports_image_input=True,
        tunable=None,
        base_model_details=SimpleNamespace(tunable=False),
        supports_lora=True,
        supports_serverless=False,
        context_length=262_144,
        training_context_length=131_072,
        model_extra={"supervisedLoraTunable": True, "useTrainingV2": True},
    )

    class Models:
        def list(self, **kwargs: object) -> list[SimpleNamespace]:
            return [model]

        def get(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return model

    exit_code, payload = cli.run_fireworks_preflight(
        "Qwen3.5-9B", str(env_file), client=SimpleNamespace(models=Models())
    )

    assert exit_code == 2
    assert payload["status"] == "ineligible"
    assert payload["training_execution_mode"] == "provider_dedicated"


def test_parser_defaults_to_accepted_provider_hypothesis() -> None:
    args = cli.build_parser().parse_args(["preflight", "fireworks"])
    assert args.hypothesis == "Qwen3-VL-8B-Instruct"
    assert args.env_file == ".env"

    args = cli.build_parser().parse_args(["run", "prepare"])
    assert args.workers == 8
    assert args.runs_root == "runs"
    assert args.config == "experiments/edugraph-20260913-qwen35-9b-local-ddp-v1.json"


def test_main_prints_payload_and_preserves_preflight_exit_code(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "run_fireworks_preflight",
        lambda hypothesis, env_file: (2, {"status": "ineligible", "hypothesis": hypothesis}),
    )

    assert cli.main(["preflight", "fireworks"]) == 2
    assert '"status": "ineligible"' in capsys.readouterr().out


def test_main_reports_safe_known_and_unexpected_errors(monkeypatch, capsys) -> None:
    def known(*args: object) -> object:
        raise ModelResolutionError("missing model")

    monkeypatch.setattr(cli, "run_fireworks_preflight", known)
    assert cli.main(["preflight", "fireworks"]) == 2
    assert "missing model" in capsys.readouterr().err

    def unexpected(*args: object) -> object:
        raise RuntimeError("secret-value-must-not-be-printed")

    monkeypatch.setattr(cli, "run_fireworks_preflight", unexpected)
    assert cli.main(["preflight", "fireworks"]) == 1
    error = capsys.readouterr().err
    assert "RuntimeError" in error
    assert "secret-value" not in error


def test_main_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_code_commit_requires_a_clean_worktree(monkeypatch, tmp_path: Path) -> None:
    responses = iter([SimpleNamespace(stdout=""), SimpleNamespace(stdout="a" * 40 + "\n")])
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: next(responses))
    assert cli._code_commit(tmp_path) == "a" * 40

    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M tracked.py\n"),
    )
    with pytest.raises(TrainingConfigError, match="clean committed"):
        cli._code_commit(tmp_path)


def test_run_prepare_passes_a_pinned_commit_to_preparation(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    class Prepared:
        def to_mapping(self) -> dict[str, object]:
            return {"run_id": "run-1"}

    def fake_prepare(path: Path, **kwargs: object) -> Prepared:
        observed.update({"path": path, **kwargs})
        return Prepared()

    monkeypatch.setattr(cli, "prepare_run", fake_prepare)
    exit_code, payload = cli.run_prepare(
        "experiment.json",
        str(tmp_path / "runs"),
        3,
        repo_root=tmp_path,
        code_commit="a" * 40,
    )

    assert exit_code == 0
    assert payload == {"run_id": "run-1"}
    assert observed["code_commit"] == "a" * 40
    assert observed["workers"] == 3


@pytest.mark.parametrize(
    "config_path",
    sorted(
        path for path in (Path(__file__).resolve().parents[1] / "experiments").glob("edugraph-20260823-*.json")
        if not path.name.endswith(".record.json")
    ),
    ids=lambda path: path.stem,
)
def test_historical_preparation_reports_version_mismatch_before_loading_data(
    config_path: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "_code_commit", lambda root: "a" * 40)
    monkeypatch.setattr(
        training,
        "load_released_splits",
        lambda *args, **kwargs: pytest.fail("version mismatch must precede dataset access"),
    )
    runs_root = tmp_path / "runs"
    assert cli.main([
        "run", "prepare", "--config", str(config_path), "--runs-root", str(runs_root)
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["message"] == "expected edugraph-py 0.21.0, found 0.26.0"
    assert not runs_root.exists()


def test_run_launch_requires_exact_confirmation_and_records_external_ids(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"run_id":"run-1","code_commit":"' + "a" * 40 + '"}',
        encoding="utf-8",
    )
    artifact = tmp_path / "train.jsonl"
    artifact.write_text("{}\n{}\n{}\n", encoding="utf-8")
    uploads = tuple(
        DatasetUploadSpec(
            account_id="account-1",
            dataset_id=f"{split}-1",
            display_name=split,
            split=split,
            path=artifact,
            example_count=3,
            sha256="a" * 64,
        )
        for split in ("train", "validation")
    )
    job_spec = SupervisedFineTuningSpec(
        account_id="account-1",
        job_id="job-1",
        display_name="run-1",
        base_model="accounts/fireworks/models/qwen3-vl-8b-instruct",
        output_model_id="model-1",
        training_dataset=uploads[0].resource_name,
        evaluation_dataset=uploads[1].resource_name,
        epochs=1,
        lora_rank=8,
        early_stop=False,
        eval_auto_carveout=False,
    )
    monkeypatch.setattr(cli, "launch_specs", lambda path: (uploads, job_spec))

    class Provider:
        def __init__(self, client: object) -> None:
            self.client = client

        def preflight_model(self, hypothesis: str) -> SimpleNamespace:
            return SimpleNamespace(
                status=EligibilityStatus.ELIGIBLE,
                capabilities=SimpleNamespace(
                    identity=ModelIdentity("fireworks", job_spec.base_model)
                ),
            )

        def validate_launch_targets(
            self, *args: object, **kwargs: object
        ) -> frozenset[str]:
            return frozenset()

        def upload_dataset(
            self, upload: DatasetUploadSpec, *, create: bool = True
        ) -> ProviderDatasetRecord:
            assert create is True
            return ProviderDatasetRecord(upload.resource_name, "READY", 3)

        def launch_supervised_fine_tuning(
            self, spec: SupervisedFineTuningSpec
        ) -> TrainingJobRecord:
            return TrainingJobRecord(
                name="accounts/account-1/supervisedFineTuningJobs/job-1",
                state="JOB_STATE_CREATING",
                base_model=spec.base_model,
                output_model=spec.output_model_name,
                dataset=spec.training_dataset,
                evaluation_dataset=spec.evaluation_dataset,
                epochs=1,
                lora_rank=8,
                learning_rate=0.0001,
                max_context_length=32768,
                estimated_cost=3.0,
                status_code="OK",
                status_message="",
            )

        def get_supervised_fine_tuning_job(
            self, spec: SupervisedFineTuningSpec
        ) -> TrainingJobRecord:
            raise AssertionError("new launch must not attempt to resume a job")

    monkeypatch.setattr(cli, "FireworksTrainingProvider", Provider)

    with pytest.raises(TrainingLaunchError, match="confirmation"):
        cli.run_launch(
            str(manifest), "wrong", ".env", client=object(), code_commit="a" * 40
        )

    with pytest.raises(TrainingLaunchError, match="same clean code commit"):
        cli.run_launch(
            str(manifest), "run-1", ".env", client=object(), code_commit="b" * 40
        )

    exit_code, payload = cli.run_launch(
        str(manifest), "run-1", ".env", client=object(), code_commit="a" * 40
    )

    assert exit_code == 0
    assert payload["status"] == "launched"
    assert payload["job"]["estimated_cost"] == 3.0
    assert (tmp_path / "launch-record.json").is_file()


def test_run_launch_resumes_only_resources_checkpointed_by_same_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"run_id":"run-1","code_commit":"' + "a" * 40 + '"}',
        encoding="utf-8",
    )
    upload = DatasetUploadSpec(
        account_id="account-1",
        dataset_id="train-1",
        display_name="run-1 train",
        split="train",
        path=tmp_path / "train.jsonl",
        example_count=3,
        sha256="a" * 64,
    )
    upload.path.write_text("{}\n{}\n{}\n", encoding="utf-8")
    validation = DatasetUploadSpec(
        account_id="account-1",
        dataset_id="validation-1",
        display_name="run-1 validation",
        split="validation",
        path=upload.path,
        example_count=3,
        sha256="a" * 64,
    )
    job_spec = SupervisedFineTuningSpec(
        account_id="account-1",
        job_id="job-1",
        display_name="run-1",
        base_model="accounts/fireworks/models/qwen3-vl-8b-instruct",
        output_model_id="model-1",
        training_dataset=upload.resource_name,
        evaluation_dataset=validation.resource_name,
        epochs=1,
        lora_rank=8,
        early_stop=False,
        eval_auto_carveout=False,
    )
    monkeypatch.setattr(cli, "launch_specs", lambda path: ((upload, validation), job_spec))
    manifest_hash = cli.file_sha256(manifest)
    (tmp_path / "launch-record.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "manifest_sha256": manifest_hash,
                "status": "dataset_upload_pending",
                "pending_dataset": upload.resource_name,
                "datasets": [],
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    class Provider:
        def __init__(self, client: object) -> None:
            pass

        def preflight_model(self, hypothesis: str) -> SimpleNamespace:
            return SimpleNamespace(
                status=EligibilityStatus.ELIGIBLE,
                capabilities=SimpleNamespace(identity=ModelIdentity("fireworks", job_spec.base_model)),
            )

        def validate_launch_targets(
            self, *args: object, **kwargs: object
        ) -> frozenset[str]:
            observed["allowed"] = kwargs["allowed_existing"]
            return frozenset({upload.resource_name})

        def upload_dataset(
            self, spec: DatasetUploadSpec, *, create: bool = True
        ) -> ProviderDatasetRecord:
            observed[spec.split] = create
            return ProviderDatasetRecord(spec.resource_name, "READY", 3)

        def launch_supervised_fine_tuning(
            self, spec: SupervisedFineTuningSpec
        ) -> TrainingJobRecord:
            return TrainingJobRecord(
                name="accounts/account-1/supervisedFineTuningJobs/job-1",
                state="JOB_STATE_CREATING",
                base_model=spec.base_model,
                output_model=spec.output_model_name,
                dataset=spec.training_dataset,
                evaluation_dataset=spec.evaluation_dataset,
                epochs=1,
                lora_rank=8,
                learning_rate=None,
                max_context_length=None,
                estimated_cost=None,
                status_code="OK",
                status_message="",
            )

        def get_supervised_fine_tuning_job(
            self, spec: SupervisedFineTuningSpec
        ) -> TrainingJobRecord:
            raise AssertionError("job was not pending")

    monkeypatch.setattr(cli, "FireworksTrainingProvider", Provider)
    exit_code, payload = cli.run_launch(
        str(manifest), "run-1", ".env", client=object(), code_commit="a" * 40
    )

    assert exit_code == 0
    assert payload["status"] == "launched"
    assert observed["allowed"] == frozenset({upload.resource_name})
    assert observed["train"] is False
    assert observed["validation"] is True


def test_main_routes_prepare_and_launch_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "run_prepare",
        lambda config, runs_root, workers: (0, {"status": "prepared"}),
    )
    assert cli.main(["run", "prepare"]) == 0
    assert "prepared" in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "run_launch",
        lambda manifest, confirmation, env_file: (0, {"status": "launched"}),
    )
    assert (
        cli.main(
            [
                "run",
                "launch",
                "--manifest",
                "manifest.json",
                "--confirm-run-id",
                "run-1",
            ]
        )
        == 0
    )
    assert "launched" in capsys.readouterr().out


def vertex_config() -> dict[str, object]:
    return {
        "provider_id": "gcp_vertex_ai",
        "training_execution_mode": "serverless",
        "job_id": "edugraph-vertex-smoke",
        "code_commit": "a" * 40,
        "project_id": "edugraph-prod",
        "location": "europe-west4",
        "display_name": "EduGraph Vertex smoke",
        "output_uri": "gs://edugraph-training/runs/vertex-smoke",
        "service_account": "vertex-training@edugraph-prod.iam.gserviceaccount.com",
        "container": {
            "image_uri": "example.com/trainer@sha256:" + "b" * 64,
            "command": [],
            "args": [],
            "environment": {},
        },
        "compute": {
            "machine_type": "g2-standard-12",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,
            "replica_count": 1,
            "boot_disk_type": "pd-ssd",
            "boot_disk_size_gb": 500,
        },
        "scheduling": {
            "strategy": "FLEX_START",
            "timeout_seconds": 14400,
            "max_wait_seconds": 7200,
        },
        "labels": {"workload": "vlm-training"},
    }


class FakeVertexClient:
    def __init__(self, existing: list[object] | None = None) -> None:
        self.existing = existing or []
        self.created = 0

    def list_custom_jobs(self, **kwargs: object) -> list[object]:
        return self.existing

    def create_custom_job(self, **kwargs: object) -> SimpleNamespace:
        self.created += 1
        return SimpleNamespace(
            name="projects/edugraph-prod/locations/europe-west4/customJobs/123",
            display_name="EduGraph Vertex smoke",
            state=SimpleNamespace(name="JOB_STATE_PENDING"),
            error=SimpleNamespace(code=0, message=""),
        )


def test_vertex_render_is_offline_and_exposes_exact_regional_request(tmp_path: Path) -> None:
    path = tmp_path / "vertex.json"
    path.write_text(json.dumps(vertex_config()), encoding="utf-8")

    exit_code, payload = cli.run_vertex_render(str(path))

    assert exit_code == 0
    assert payload["status"] == "validated"
    assert payload["training_execution_mode"] == "serverless"
    assert payload["api_endpoint"] == "europe-west4-aiplatform.googleapis.com"
    assert payload["request"]["parent"].endswith("locations/europe-west4")  # type: ignore[index,union-attr]


def test_vertex_launch_requires_confirmation_and_pinned_commit(tmp_path: Path) -> None:
    path = tmp_path / "vertex.json"
    path.write_text(json.dumps(vertex_config()), encoding="utf-8")
    client = FakeVertexClient()

    with pytest.raises(VertexLaunchError, match="confirmation"):
        cli.run_vertex_launch(
            str(path), "wrong", ".env", str(tmp_path / "runs"), client=client,
            code_commit="a" * 40,
        )
    with pytest.raises(VertexLaunchError, match="same clean code commit"):
        cli.run_vertex_launch(
            str(path), "edugraph-vertex-smoke", ".env", str(tmp_path / "runs"),
            client=client, code_commit="c" * 40,
        )

    exit_code, payload = cli.run_vertex_launch(
        str(path), "edugraph-vertex-smoke", ".env", str(tmp_path / "runs"),
        client=client, code_commit="a" * 40,
    )
    repeated_code, repeated = cli.run_vertex_launch(
        str(path), "edugraph-vertex-smoke", ".env", str(tmp_path / "runs"),
        client=client, code_commit="a" * 40,
    )

    assert exit_code == repeated_code == 0
    assert payload["status"] == repeated["status"] == "launched"
    assert payload["job"]["name"].endswith("customJobs/123")  # type: ignore[index,union-attr]
    assert client.created == 1
    assert Path(payload["launch_record_path"]).is_file()  # type: ignore[arg-type]


def test_vertex_launch_collision_and_pending_resume_are_fail_safe(tmp_path: Path) -> None:
    path = tmp_path / "vertex.json"
    path.write_text(json.dumps(vertex_config()), encoding="utf-8")
    existing_job = SimpleNamespace(
        name="projects/edugraph-prod/locations/europe-west4/customJobs/456",
        display_name="EduGraph Vertex smoke",
        state="JOB_STATE_RUNNING",
        error=None,
    )
    client = FakeVertexClient([existing_job])
    records = tmp_path / "runs"

    with pytest.raises(VertexLaunchError, match="already exists"):
        cli.run_vertex_launch(
            str(path), "edugraph-vertex-smoke", ".env", str(records),
            client=client, code_commit="a" * 40,
        )

    record_path = records / "edugraph-vertex-smoke" / "vertex-launch-record.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "job_id": "edugraph-vertex-smoke",
                "config_sha256": cli.file_sha256(path),
                "status": "job_launch_pending",
            }
        ),
        encoding="utf-8",
    )
    _, resumed = cli.run_vertex_launch(
        str(path), "edugraph-vertex-smoke", ".env", str(records),
        client=client, code_commit="a" * 40,
    )
    assert resumed["job"]["name"].endswith("customJobs/456")  # type: ignore[index,union-attr]
    assert client.created == 0

    record_path.unlink()
    client.existing.append(existing_job)
    with pytest.raises(VertexLaunchError, match="multiple"):
        cli.run_vertex_launch(
            str(path), "edugraph-vertex-smoke", ".env", str(records),
            client=client, code_commit="a" * 40,
        )


def test_vertex_stage_requires_confirmation_and_records_retry_identity(
    monkeypatch, tmp_path: Path
) -> None:
    applied: list[object] = []

    class Plan:
        run_id = "vertex-smoke"
        runtime_manifest_uri = "gs://bucket-1/runtime.json"
        runtime_manifest_sha256 = "d" * 64

        def to_mapping(self) -> dict[str, object]:
            return {"run_id": self.run_id, "artifacts": []}

    plan = Plan()
    monkeypatch.setattr(cli, "build_vertex_staging_plan", lambda path, bucket: plan)
    monkeypatch.setattr(cli, "GcsArtifactStore", lambda client: client)
    monkeypatch.setattr(cli, "stage_vertex_run", lambda value, writer: applied.append(writer))

    with pytest.raises(VertexArtifactError, match="confirmation"):
        cli.run_vertex_stage(
            "manifest.json", "gs://bucket-1", "wrong", str(tmp_path), client=object()
        )

    client = object()
    exit_code, payload = cli.run_vertex_stage(
        "manifest.json", "gs://bucket-1", "vertex-smoke", str(tmp_path), client=client
    )
    repeated_code, repeated = cli.run_vertex_stage(
        "manifest.json", "gs://bucket-1", "vertex-smoke", str(tmp_path), client=client
    )

    assert exit_code == repeated_code == 0
    assert payload["status"] == repeated["status"] == "staged"
    assert applied == [client]


def test_vertex_configure_requires_prepared_commit_and_writes_validated_file(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"code_commit": "a" * 40}), encoding="utf-8")
    staging = tmp_path / "staging.json"
    staging.write_text("{}", encoding="utf-8")
    config = vertex_config()
    observed: dict[str, object] = {}

    def build(*args: object, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return config

    monkeypatch.setattr(cli, "build_vertex_job_config", build)

    with pytest.raises(VertexArtifactError, match="same clean commit"):
        cli.run_vertex_configure(
            str(manifest), str(staging), config["container"]["image_uri"],  # type: ignore[index]
            str(tmp_path / "job.json"), code_commit="b" * 40,
        )

    exit_code, payload = cli.run_vertex_configure(
        str(manifest), str(staging), config["container"]["image_uri"],  # type: ignore[index]
        str(tmp_path / "job.json"),
        diagnostic_one_batch=True,
        disable_native_jit=True,
        code_commit="a" * 40,
    )
    assert exit_code == 0
    assert payload["status"] == "configured"
    assert (tmp_path / "job.json").is_file()
    assert observed == {"diagnostic_one_batch": True, "disable_native_jit": True}


def test_main_routes_vertex_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli, "run_vertex_render", lambda config: (0, {"status": "validated"})
    )
    assert cli.main(["vertex", "render", "--config", "vertex.json"]) == 0
    assert "validated" in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "run_vertex_launch",
        lambda config, confirmation, env_file, records_root: (0, {"status": "launched"}),
    )
    assert cli.main(
        [
            "vertex", "launch", "--config", "vertex.json",
            "--confirm-job-id", "run-1",
        ]
    ) == 0
    assert "launched" in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "run_vertex_stage",
        lambda manifest, bucket, confirmation, records: (0, {"status": "staged"}),
    )
    assert cli.main(
        [
            "vertex", "stage", "--manifest", "manifest.json",
            "--bucket-uri", "gs://bucket-1", "--confirm-run-id", "run-1",
        ]
    ) == 0
    assert "staged" in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "run_vertex_configure",
        lambda manifest, staging, image, output, **kwargs: (
            0,
            {
                "status": "configured",
                "diagnostic": kwargs["diagnostic_one_batch"],
                "native_jit_disabled": kwargs["disable_native_jit"],
            },
        ),
    )
    assert cli.main(
        [
            "vertex", "configure", "--manifest", "manifest.json",
            "--staging-record", "staging.json", "--image-uri",
            "example.com/vlm@sha256:" + "a" * 64, "--output", "job.json",
            "--diagnostic-one-batch",
            "--disable-native-jit",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "configured" in output
    assert '"diagnostic": true' in output
    assert '"native_jit_disabled": true' in output
