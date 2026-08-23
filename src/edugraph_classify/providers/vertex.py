"""Thin Google Vertex AI serverless custom-training adapter."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from ..contracts import ExecutionMode


class VertexConfigError(ValueError):
    """Raised when a Vertex custom-job configuration is unsafe or incomplete."""


class VertexLaunchError(RuntimeError):
    """Raised when a guarded Vertex custom-job launch cannot proceed safely."""


class SchedulingStrategy(StrEnum):
    """Vertex scheduling strategies supported by the custom-job API."""

    STANDARD = "STANDARD"
    SPOT = "SPOT"
    FLEX_START = "FLEX_START"


class JobServiceClient(Protocol):
    def list_custom_jobs(self, **kwargs: object) -> Iterable[object]: ...

    def create_custom_job(self, **kwargs: object) -> object: ...


_RESOURCE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_PROJECT_ID = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_LOCATION = re.compile(r"^[a-z]+-[a-z]+[0-9]+$")
_IMAGE_DIGEST = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
_GCS_URI = re.compile(r"^gs://[a-z0-9][a-z0-9._-]{1,220}/[^\s]+$")
_SERVICE_ACCOUNT = re.compile(r"^[a-z0-9-]+@[a-z0-9.-]+\.iam\.gserviceaccount\.com$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_LABEL = re.compile(r"^[a-z0-9_-]{1,63}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SECRET_NAME = re.compile(r"(?:TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL)")


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise VertexConfigError(f"{field_name} must be an object")
    return value


def _require_exact_keys(
    payload: Mapping[str, object], expected: set[str], field_name: str
) -> None:
    if set(payload) != expected:
        raise VertexConfigError(f"{field_name} must contain exactly {sorted(expected)}")


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise VertexConfigError(f"{field_name} must be a non-empty trimmed string")
    return value


def _string_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise VertexConfigError(f"{field_name} must be an array of strings")
    return tuple(_require_string(item, f"{field_name} item") for item in value)


def _string_pairs(
    value: object,
    field_name: str,
    *,
    key_pattern: re.Pattern[str],
) -> tuple[tuple[str, str], ...]:
    mapping = _require_mapping(value, field_name)
    pairs: list[tuple[str, str]] = []
    for raw_key, raw_value in mapping.items():
        key = _require_string(raw_key, f"{field_name} key")
        item = _require_string(raw_value, f"{field_name}.{key}")
        if not key_pattern.fullmatch(key):
            raise VertexConfigError(f"{field_name} contains an invalid key: {key}")
        pairs.append((key, item))
    return tuple(sorted(pairs))


@dataclass(frozen=True, slots=True)
class VertexContainerSpec:
    """An immutable custom training image and its non-secret invocation."""

    image_uri: str
    command: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not _IMAGE_DIGEST.fullmatch(self.image_uri):
            raise VertexConfigError(
                "container.image_uri must use an immutable @sha256 digest"
            )
        for name, _ in self.environment:
            if not _ENVIRONMENT_NAME.fullmatch(name):
                raise VertexConfigError(f"container.environment has an invalid name: {name}")
            if _SECRET_NAME.search(name):
                raise VertexConfigError(
                    "credentials must not be embedded in the custom-job configuration"
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> VertexContainerSpec:
        _require_exact_keys(
            payload,
            {"image_uri", "command", "args", "environment"},
            "container",
        )
        return cls(
            image_uri=_require_string(payload["image_uri"], "container.image_uri"),
            command=_string_sequence(payload["command"], "container.command"),
            args=_string_sequence(payload["args"], "container.args"),
            environment=_string_pairs(
                payload["environment"],
                "container.environment",
                key_pattern=_ENVIRONMENT_NAME,
            ),
        )


@dataclass(frozen=True, slots=True)
class VertexComputeSpec:
    """One homogeneous Vertex worker pool."""

    machine_type: str
    accelerator_type: str
    accelerator_count: int
    replica_count: int
    boot_disk_type: str
    boot_disk_size_gb: int

    def __post_init__(self) -> None:
        for field_name in ("machine_type", "accelerator_type", "boot_disk_type"):
            _require_string(getattr(self, field_name), f"compute.{field_name}")
        if not self.accelerator_type.startswith("NVIDIA_"):
            raise VertexConfigError("compute.accelerator_type must be a Vertex NVIDIA enum")
        if self.accelerator_count < 1 or self.replica_count < 1:
            raise VertexConfigError("compute accelerator and replica counts must be positive")
        if self.boot_disk_size_gb < 100:
            raise VertexConfigError("compute.boot_disk_size_gb must be at least 100")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> VertexComputeSpec:
        expected = {
            "machine_type",
            "accelerator_type",
            "accelerator_count",
            "replica_count",
            "boot_disk_type",
            "boot_disk_size_gb",
        }
        _require_exact_keys(payload, expected, "compute")
        integers: dict[str, int] = {}
        for field_name in ("accelerator_count", "replica_count", "boot_disk_size_gb"):
            value = payload[field_name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise VertexConfigError(f"compute.{field_name} must be an integer")
            integers[field_name] = value
        return cls(
            machine_type=_require_string(payload["machine_type"], "compute.machine_type"),
            accelerator_type=_require_string(
                payload["accelerator_type"], "compute.accelerator_type"
            ),
            accelerator_count=integers["accelerator_count"],
            replica_count=integers["replica_count"],
            boot_disk_type=_require_string(
                payload["boot_disk_type"], "compute.boot_disk_type"
            ),
            boot_disk_size_gb=integers["boot_disk_size_gb"],
        )


@dataclass(frozen=True, slots=True)
class VertexSchedulingSpec:
    """Bounded execution and provisioning wait behavior."""

    strategy: SchedulingStrategy
    timeout_seconds: int
    max_wait_seconds: int | None

    def __post_init__(self) -> None:
        if self.timeout_seconds < 1:
            raise VertexConfigError("scheduling.timeout_seconds must be positive")
        if self.strategy is SchedulingStrategy.FLEX_START:
            if self.max_wait_seconds is None or self.max_wait_seconds < 1:
                raise VertexConfigError(
                    "FLEX_START requires a positive scheduling.max_wait_seconds"
                )
        elif self.max_wait_seconds is not None:
            raise VertexConfigError(
                "scheduling.max_wait_seconds is only valid with FLEX_START"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> VertexSchedulingSpec:
        _require_exact_keys(
            payload,
            {"strategy", "timeout_seconds", "max_wait_seconds"},
            "scheduling",
        )
        raw_strategy = _require_string(payload["strategy"], "scheduling.strategy")
        try:
            strategy = SchedulingStrategy(raw_strategy)
        except ValueError as error:
            raise VertexConfigError(
                f"unsupported scheduling.strategy: {raw_strategy}"
            ) from error
        timeout = payload["timeout_seconds"]
        max_wait = payload["max_wait_seconds"]
        if not isinstance(timeout, int) or isinstance(timeout, bool):
            raise VertexConfigError("scheduling.timeout_seconds must be an integer")
        if max_wait is not None and (
            not isinstance(max_wait, int) or isinstance(max_wait, bool)
        ):
            raise VertexConfigError("scheduling.max_wait_seconds must be an integer or null")
        return cls(strategy=strategy, timeout_seconds=timeout, max_wait_seconds=max_wait)


@dataclass(frozen=True, slots=True)
class VertexCustomJobSpec:
    """Fully resolved, reproducible Vertex CustomJob launch configuration."""

    job_id: str
    code_commit: str
    project_id: str
    location: str
    display_name: str
    output_uri: str
    service_account: str
    container: VertexContainerSpec
    compute: VertexComputeSpec
    scheduling: VertexSchedulingSpec
    labels: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not _RESOURCE_ID.fullmatch(self.job_id):
            raise VertexConfigError("job_id is not a valid stable resource identity")
        if not _COMMIT.fullmatch(self.code_commit):
            raise VertexConfigError("code_commit must be a full lowercase Git commit")
        if not _PROJECT_ID.fullmatch(self.project_id):
            raise VertexConfigError("project_id is not a valid Google Cloud project ID")
        if not _LOCATION.fullmatch(self.location):
            raise VertexConfigError("location must be a regional Vertex AI location")
        _require_string(self.display_name, "display_name")
        if len(self.display_name) > 128:
            raise VertexConfigError("display_name must not exceed 128 characters")
        if not _GCS_URI.fullmatch(self.output_uri):
            raise VertexConfigError("output_uri must be a non-root gs:// URI")
        if not _SERVICE_ACCOUNT.fullmatch(self.service_account):
            raise VertexConfigError("service_account must be a service-account email")
        for key, value in self.labels:
            if not _LABEL.fullmatch(key) or not _LABEL.fullmatch(value):
                raise VertexConfigError("labels must use lowercase Google Cloud label syntax")
        if any(key == "edugraph_job_id" for key, _ in self.labels):
            raise VertexConfigError("edugraph_job_id is reserved by the adapter")

    @property
    def parent(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}"

    @property
    def api_endpoint(self) -> str:
        return f"{self.location}-aiplatform.googleapis.com"

    @classmethod
    def from_mapping(cls, payload: object) -> VertexCustomJobSpec:
        payload = _require_mapping(payload, "Vertex job configuration")
        _require_exact_keys(
            payload,
            {
                "provider_id",
                "training_execution_mode",
                "job_id",
                "code_commit",
                "project_id",
                "location",
                "display_name",
                "output_uri",
                "service_account",
                "container",
                "compute",
                "scheduling",
                "labels",
            },
            "Vertex job configuration",
        )
        if payload["provider_id"] != VertexTrainingProvider.provider_id:
            raise VertexConfigError("provider_id must be gcp_vertex_ai")
        if payload["training_execution_mode"] != ExecutionMode.SERVERLESS:
            raise VertexConfigError("training_execution_mode must be serverless")
        return cls(
            job_id=_require_string(payload["job_id"], "job_id"),
            code_commit=_require_string(payload["code_commit"], "code_commit"),
            project_id=_require_string(payload["project_id"], "project_id"),
            location=_require_string(payload["location"], "location"),
            display_name=_require_string(payload["display_name"], "display_name"),
            output_uri=_require_string(payload["output_uri"], "output_uri"),
            service_account=_require_string(payload["service_account"], "service_account"),
            container=VertexContainerSpec.from_mapping(
                _require_mapping(payload["container"], "container")
            ),
            compute=VertexComputeSpec.from_mapping(
                _require_mapping(payload["compute"], "compute")
            ),
            scheduling=VertexSchedulingSpec.from_mapping(
                _require_mapping(payload["scheduling"], "scheduling")
            ),
            labels=_string_pairs(payload["labels"], "labels", key_pattern=_LABEL),
        )


@dataclass(frozen=True, slots=True)
class VertexJobRecord:
    """Non-SDK record retained after a Vertex job is resolved or created."""

    name: str
    display_name: str | None
    state: str | None
    error_code: int | None
    error_message: str | None

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def load_vertex_job_config(path: Path) -> VertexCustomJobSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return VertexCustomJobSpec.from_mapping(_require_mapping(payload, "configuration"))


def _enum_name(value: object) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


class VertexTrainingProvider:
    """Render and submit Vertex AI serverless custom-container training jobs."""

    provider_id = "gcp_vertex_ai"
    training_execution_mode = ExecutionMode.SERVERLESS

    def __init__(self, client: JobServiceClient | None = None) -> None:
        self._client = client

    def _require_client(self) -> JobServiceClient:
        if self._client is None:
            raise VertexLaunchError("a Vertex JobService client is required for cloud operations")
        return self._client

    @staticmethod
    def _labels(spec: VertexCustomJobSpec) -> dict[str, str]:
        return {**dict(spec.labels), "edugraph_job_id": spec.job_id}

    def render_custom_job(self, spec: VertexCustomJobSpec) -> dict[str, object]:
        scheduling: dict[str, object] = {
            "strategy": spec.scheduling.strategy.value,
            "timeout": f"{spec.scheduling.timeout_seconds}s",
        }
        if spec.scheduling.max_wait_seconds is not None:
            scheduling["max_wait_duration"] = (
                f"{spec.scheduling.max_wait_seconds}s"
            )
        container: dict[str, object] = {
            "image_uri": spec.container.image_uri,
            "command": list(spec.container.command),
            "args": list(spec.container.args),
        }
        if spec.container.environment:
            container["env"] = [
                {"name": name, "value": value}
                for name, value in spec.container.environment
            ]
        custom_job = {
            "display_name": spec.display_name,
            "labels": self._labels(spec),
            "job_spec": {
                "service_account": spec.service_account,
                "base_output_directory": {"output_uri_prefix": spec.output_uri},
                "scheduling": scheduling,
                "worker_pool_specs": [
                    {
                        "machine_spec": {
                            "machine_type": spec.compute.machine_type,
                            "accelerator_type": spec.compute.accelerator_type,
                            "accelerator_count": spec.compute.accelerator_count,
                        },
                        "replica_count": spec.compute.replica_count,
                        "disk_spec": {
                            "boot_disk_type": spec.compute.boot_disk_type,
                            "boot_disk_size_gb": spec.compute.boot_disk_size_gb,
                        },
                        "container_spec": container,
                    }
                ],
            },
        }
        return {"parent": spec.parent, "custom_job": custom_job}

    def find_custom_jobs(self, spec: VertexCustomJobSpec) -> tuple[VertexJobRecord, ...]:
        observed = self._require_client().list_custom_jobs(
            request={
                "parent": spec.parent,
                "filter": f"labels.edugraph_job_id={spec.job_id}",
            }
        )
        return tuple(self._record(job) for job in observed)

    def launch_custom_job(self, spec: VertexCustomJobSpec) -> VertexJobRecord:
        job = self._require_client().create_custom_job(
            request=self.render_custom_job(spec)
        )
        return self._record(job)

    @staticmethod
    def _record(job: object) -> VertexJobRecord:
        error = getattr(job, "error", None)
        code = getattr(error, "code", None)
        return VertexJobRecord(
            name=getattr(job, "name", ""),
            display_name=getattr(job, "display_name", None),
            state=_enum_name(getattr(job, "state", None)),
            error_code=code if isinstance(code, int) else None,
            error_message=getattr(error, "message", None),
        )
