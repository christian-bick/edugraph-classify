from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.cloud import aiplatform_v1

from edugraph_classify.contracts import ExecutionMode
from edugraph_classify.providers import ServerlessTrainingProvider
from edugraph_classify.providers.vertex import (
    SchedulingStrategy,
    VertexComputeSpec,
    VertexConfigError,
    VertexContainerSpec,
    VertexCustomJobSpec,
    VertexLaunchError,
    VertexSchedulingSpec,
    VertexTrainingProvider,
    load_vertex_job_config,
)


def config_mapping() -> dict[str, object]:
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
            "image_uri": (
                "europe-west4-docker.pkg.dev/edugraph-prod/training/vlm"
                "@sha256:" + "b" * 64
            ),
            "command": ["python", "-m", "trainer"],
            "args": ["--run-manifest", "gs://edugraph-training/runs/manifest.json"],
            "environment": {"HF_HOME": "/tmp/huggingface"},
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
        "labels": {"environment": "smoke", "workload": "vlm-training"},
    }


def job_spec() -> VertexCustomJobSpec:
    return VertexCustomJobSpec.from_mapping(config_mapping())


class FakeJobs:
    def __init__(self, existing: list[object] | None = None) -> None:
        self.existing = existing or []
        self.list_request: object | None = None
        self.create_request: object | None = None

    def list_custom_jobs(self, request: object | None = None) -> list[object]:
        self.list_request = request
        return self.existing

    def create_custom_job(self, request: object | None = None) -> object:
        self.create_request = request
        return SimpleNamespace(
            name="projects/edugraph-prod/locations/europe-west4/customJobs/123",
            display_name="EduGraph Vertex smoke",
            state=SimpleNamespace(name="JOB_STATE_PENDING"),
            error=SimpleNamespace(code=0, message=""),
        )


def test_vertex_config_renders_an_sdk_valid_regional_custom_job(tmp_path: Path) -> None:
    path = tmp_path / "vertex.json"
    path.write_text(json.dumps(config_mapping()), encoding="utf-8")
    spec = load_vertex_job_config(path)
    client = FakeJobs()
    adapter = VertexTrainingProvider(client)

    request = adapter.render_custom_job(spec)
    custom_job = request["custom_job"]
    parsed = aiplatform_v1.CustomJob(mapping=custom_job)

    assert isinstance(adapter, ServerlessTrainingProvider)
    assert adapter.training_execution_mode is ExecutionMode.SERVERLESS
    assert spec.api_endpoint == "europe-west4-aiplatform.googleapis.com"
    assert request["parent"] == "projects/edugraph-prod/locations/europe-west4"
    assert parsed.labels["edugraph_job_id"] == spec.job_id
    worker = parsed.job_spec.worker_pool_specs[0]
    assert worker.machine_spec.accelerator_type.name == "NVIDIA_L4"
    assert worker.disk_spec.boot_disk_size_gb == 500
    assert parsed.job_spec.scheduling.strategy.name == "FLEX_START"
    assert parsed.job_spec.scheduling.timeout.seconds == 14400
    assert parsed.job_spec.scheduling.max_wait_duration.seconds == 7200


def test_vertex_adapter_finds_and_launches_jobs_without_sdk_types() -> None:
    client = FakeJobs(
        [
            SimpleNamespace(
                name="existing",
                display_name="Existing",
                state="JOB_STATE_SUCCEEDED",
                error=SimpleNamespace(code=7, message="done"),
            )
        ]
    )
    adapter = VertexTrainingProvider(client)
    spec = job_spec()

    found = adapter.find_custom_jobs(spec)
    launched = adapter.launch_custom_job(spec)

    assert client.list_request == {
        "parent": spec.parent,
        "filter": "labels.edugraph_job_id=edugraph-vertex-smoke",
    }
    assert found[0].to_mapping() == {
        "name": "existing",
        "display_name": "Existing",
        "state": "JOB_STATE_SUCCEEDED",
        "error_code": 7,
        "error_message": "done",
    }
    assert launched.state == "JOB_STATE_PENDING"
    assert client.create_request == adapter.render_custom_job(spec)

    with pytest.raises(VertexLaunchError, match="client is required"):
        VertexTrainingProvider().find_custom_jobs(spec)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"project_id": "INVALID"}, "project_id"),
        ({"location": "global"}, "regional"),
        ({"output_uri": "https://bucket/output"}, "gs://"),
        ({"service_account": "user@example.com"}, "service-account"),
        ({"job_id": "Invalid"}, "job_id"),
        ({"code_commit": "short"}, "code_commit"),
        ({"display_name": "x" * 129}, "128"),
        ({"labels": (("Bad", "value"),)}, "labels"),
        ({"labels": (("edugraph_job_id", "value"),)}, "reserved"),
    ],
)
def test_vertex_job_identity_and_region_validation(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(VertexConfigError, match=message):
        replace(job_spec(), **change)


def test_vertex_container_requires_digest_and_rejects_embedded_secrets() -> None:
    with pytest.raises(VertexConfigError, match="immutable"):
        replace(job_spec().container, image_uri="example.com/trainer:latest")
    with pytest.raises(VertexConfigError, match="credentials"):
        replace(job_spec().container, environment=(("HF_TOKEN", "secret"),))
    with pytest.raises(VertexConfigError, match="invalid name"):
        replace(job_spec().container, environment=(("lowercase", "value"),))


def test_vertex_compute_and_scheduling_fail_closed() -> None:
    compute = job_spec().compute
    with pytest.raises(VertexConfigError, match="NVIDIA"):
        replace(compute, accelerator_type="TPU_V5")
    with pytest.raises(VertexConfigError, match="positive"):
        replace(compute, replica_count=0)
    with pytest.raises(VertexConfigError, match="at least 100"):
        replace(compute, boot_disk_size_gb=99)

    scheduling = job_spec().scheduling
    with pytest.raises(VertexConfigError, match="requires"):
        replace(scheduling, max_wait_seconds=None)
    with pytest.raises(VertexConfigError, match="only valid"):
        VertexSchedulingSpec(SchedulingStrategy.STANDARD, 10, 10)
    with pytest.raises(VertexConfigError, match="positive"):
        replace(scheduling, timeout_seconds=0)


def test_vertex_mapping_rejects_unknown_shape_and_value_types() -> None:
    payload = config_mapping()
    payload["unexpected"] = True
    with pytest.raises(VertexConfigError, match="exactly"):
        VertexCustomJobSpec.from_mapping(payload)

    payload = config_mapping()
    payload["provider_id"] = "other"
    with pytest.raises(VertexConfigError, match="provider_id"):
        VertexCustomJobSpec.from_mapping(payload)

    payload = config_mapping()
    payload["training_execution_mode"] = "provider_dedicated"
    with pytest.raises(VertexConfigError, match="serverless"):
        VertexCustomJobSpec.from_mapping(payload)

    payload = config_mapping()
    payload["container"] = {**payload["container"], "command": "python"}  # type: ignore[dict-item]
    with pytest.raises(VertexConfigError, match="array"):
        VertexCustomJobSpec.from_mapping(payload)

    payload = config_mapping()
    payload["compute"] = {**payload["compute"], "replica_count": True}  # type: ignore[dict-item]
    with pytest.raises(VertexConfigError, match="integer"):
        VertexCustomJobSpec.from_mapping(payload)

    payload = config_mapping()
    payload["scheduling"] = {**payload["scheduling"], "strategy": "UNKNOWN"}  # type: ignore[dict-item]
    with pytest.raises(VertexConfigError, match="unsupported"):
        VertexCustomJobSpec.from_mapping(payload)


def test_standard_schedule_omits_flex_wait_and_empty_environment() -> None:
    spec = replace(
        job_spec(),
        container=VertexContainerSpec(job_spec().container.image_uri),
        scheduling=VertexSchedulingSpec(SchedulingStrategy.SPOT, 3600, None),
        labels=(),
    )
    request = VertexTrainingProvider(FakeJobs()).render_custom_job(spec)
    custom_job = request["custom_job"]
    assert isinstance(custom_job, dict)
    scheduling = custom_job["job_spec"]["scheduling"]  # type: ignore[index]
    container = custom_job["job_spec"]["worker_pool_specs"][0]["container_spec"]  # type: ignore[index]
    assert "max_wait_duration" not in scheduling
    assert "env" not in container


def test_mapping_reports_nested_shape_errors() -> None:
    payload = config_mapping()
    payload["container"] = []
    with pytest.raises(VertexConfigError, match="container must be an object"):
        VertexCustomJobSpec.from_mapping(payload)

    payload = config_mapping()
    payload["labels"] = {"valid": "UPPER"}
    with pytest.raises(VertexConfigError, match="lowercase"):
        VertexCustomJobSpec.from_mapping(payload)

    with pytest.raises(VertexConfigError, match="object"):
        VertexCustomJobSpec.from_mapping([])

    with pytest.raises(VertexConfigError, match="integer or null"):
        VertexSchedulingSpec.from_mapping(
            {"strategy": "FLEX_START", "timeout_seconds": 10, "max_wait_seconds": True}
        )

    with pytest.raises(VertexConfigError, match="integer"):
        VertexSchedulingSpec.from_mapping(
            {"strategy": "STANDARD", "timeout_seconds": True, "max_wait_seconds": None}
        )

    with pytest.raises(VertexConfigError, match="non-empty"):
        VertexComputeSpec.from_mapping(
            {
                "machine_type": "",
                "accelerator_type": "NVIDIA_L4",
                "accelerator_count": 1,
                "replica_count": 1,
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": 100,
            }
        )
