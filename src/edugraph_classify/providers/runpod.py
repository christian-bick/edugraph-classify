"""Thin REST Pod adapter; responses and errors never expose resolved secrets."""
from __future__ import annotations

import math
import re
import httpx


class RunpodClient:
    def __init__(self, api_key: str, client=None):
        self.client = client or httpx.Client(base_url="https://rest.runpod.io/v1", timeout=60)
        self.headers = {"Authorization": "Bearer " + api_key}

    def call(self, method: str, path: str, payload=None):
        try:
            response = self.client.request(method, path, headers=self.headers, json=payload)
        except httpx.HTTPError:
            raise RuntimeError("Runpod request failed; inspect Pod state before retrying") from None
        if response.status_code >= 400:
            raise RuntimeError(f"Runpod HTTP {response.status_code}; response body suppressed")
        return response.json() if response.content else {}

    @staticmethod
    def path(pod_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", pod_id):
            raise ValueError("invalid Pod ID")
        return "/pods/" + pod_id

    def list(self):
        return self.call("GET", "/pods")

    def create(self, request):
        return self.call("POST", "/pods", request)

    def get(self, pod_id):
        return self.call("GET", self.path(pod_id) + "?includeMachine=true")

    def stop(self, pod_id):
        self.call("POST", self.path(pod_id) + "/stop")

    def terminate(self, pod_id):
        self.call("DELETE", self.path(pod_id))


def public_status(pod):
    result = {key: pod.get(key) for key in ("id", "name", "desiredStatus", "costPerHr", "gpuCount")}
    result.update(imageName=pod.get("imageName", pod.get("image")),
                  secureCloud=(pod.get("machine") or {}).get("secureCloud"))
    return result


def require_secure(pod):
    if (pod.get("machine") or {}).get("secureCloud") is not True:
        raise ValueError("Pod is not verified as Secure Cloud")


def checked_hourly_rate(pod, ceiling):
    """REST examples return currency strings; reject unknown or nonfinite rates."""
    raw = pod.get("costPerHr")
    try:
        price = float(raw) if type(raw) in (str, int, float) else math.nan
    except ValueError:
        price = math.nan
    if not math.isfinite(price) or not 0 < price <= ceiling:
        raise ValueError("Pod rate is unavailable or exceeds the approved ceiling")
    return price
