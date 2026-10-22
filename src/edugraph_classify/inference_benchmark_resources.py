"""Local capacity checks for a paid inference-benchmark Pod."""

from __future__ import annotations

import csv
import math
import os
import re
import subprocess
from pathlib import Path


def parse_gpu_inventory(output: str) -> list[dict]:
    """Parse one `nvidia-smi` row per visible device, rejecting ambiguous output."""
    devices = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 2 or not row[0].strip():
            raise ValueError("GPU inventory is incomplete")
        try:
            memory_mib = float(row[1].strip())
        except ValueError as error:
            raise ValueError("GPU memory is unavailable") from error
        if not math.isfinite(memory_mib) or memory_mib <= 0:
            raise ValueError("GPU memory is unavailable")
        devices.append({"name": row[0].strip(), "memory_gib": memory_mib / 1024})
    if not devices:
        raise ValueError("no visible GPU was reported")
    return devices


def _cgroup_memory_limit() -> int | None:
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            value = path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        if value != "max":
            try:
                limit = int(value)
            except ValueError as error:
                raise ValueError("container memory limit is invalid") from error
            if limit <= 0:
                raise ValueError("container memory limit is invalid")
            return limit
    return None


def _cgroup_cpu_quota() -> float | None:
    modern = Path("/sys/fs/cgroup/cpu.max")
    try:
        raw = modern.read_text(encoding="ascii").split()
    except OSError:
        raw = []
    if raw:
        if len(raw) != 2:
            raise ValueError("container CPU quota is invalid")
        if raw[0] == "max":
            return None
        try:
            quota, period = int(raw[0]), int(raw[1])
        except ValueError as error:
            raise ValueError("container CPU quota is invalid") from error
    else:
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="ascii"))
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="ascii"))
        except OSError:
            return None
        except ValueError as error:
            raise ValueError("container CPU quota is invalid") from error
        if quota == -1:
            return None
    if quota <= 0 or period <= 0:
        raise ValueError("container CPU quota is invalid")
    return quota / period


def local_resources() -> dict:
    """Observe GPU inventory and the container's effective CPU/RAM allowance."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    devices = parse_gpu_inventory(result.stdout)
    physical_ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    if physical_ram <= 0:
        raise ValueError("host RAM is unavailable")
    memory_limit = _cgroup_memory_limit()
    available_ram = min(physical_ram, memory_limit) if memory_limit is not None else physical_ram
    if not hasattr(os, "sched_getaffinity"):
        raise ValueError("CPU affinity is unavailable")
    affinity = len(os.sched_getaffinity(0))
    if affinity <= 0:
        raise ValueError("CPU affinity is unavailable")
    quota = _cgroup_cpu_quota()
    cpus = min(float(affinity), quota) if quota is not None else float(affinity)
    return {"gpus": devices, "gpu_count": len(devices), "host_ram_bytes": available_ram,
            "host_ram_gb_decimal": available_ram / 1_000_000_000, "cpu_count": cpus}


def _gpu_name(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^nvidia\s+", "", value.strip().lower()))


def require_capacity(observed: dict, execution: dict) -> dict:
    """Fail before model work if local resources differ from the reviewed recipe.

    `minimum_ram_gib` is a legacy recipe name passed to Runpod's GB-based
    minRAMPerGPU filter, so its local comparison uses decimal GB.
    """
    devices = observed.get("gpus")
    if not isinstance(devices, list) or len(devices) != execution["gpu_count"] or observed.get("gpu_count") != len(devices):
        raise ValueError("visible GPU count differs from the benchmark recipe")
    expected_name = _gpu_name(execution["gpu_type"])
    for device in devices:
        if _gpu_name(device["name"]) != expected_name:
            raise ValueError("visible GPU type differs from the benchmark recipe")
        if device["memory_gib"] < execution["minimum_vram_gib"]:
            raise ValueError("usable GPU memory is below the benchmark minimum")
    if observed.get("host_ram_bytes", 0) < execution["minimum_ram_gib"] * 1_000_000_000:
        raise ValueError("container RAM is below the benchmark minimum")
    if observed.get("cpu_count", 0) < execution["minimum_vcpus"]:
        raise ValueError("container CPU allowance is below the benchmark minimum")
    return observed
