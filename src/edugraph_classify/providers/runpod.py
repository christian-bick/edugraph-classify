"""Thin REST Pod adapter; responses and errors never expose resolved secrets."""
from __future__ import annotations

import math
import re
import httpx


class RunpodError(RuntimeError):
    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class RunpodClient:
    def __init__(self, api_key: str, client=None):
        self.client = client or httpx.Client(base_url="https://rest.runpod.io/v1", timeout=60)
        self.headers = {"Authorization": "Bearer " + api_key}

    def call(self, method: str, path: str, payload=None):
        try:
            response = self.client.request(method, path, headers=self.headers, json=payload)
        except httpx.HTTPError:
            raise RunpodError("Runpod request failed; inspect Pod state before retrying") from None
        if response.status_code >= 400:
            raise RunpodError(f"Runpod HTTP {response.status_code}; response body suppressed", status_code=response.status_code)
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


class RunpodPodClient(RunpodClient):
    """Pod-scoped keys work on GraphQL; the REST v1 account adapter rejects them."""
    def __init__(self, api_key, client=None):
        super().__init__(api_key, client or httpx.Client(base_url="https://api.runpod.io", timeout=60))

    def graphql(self, query, pod_id, field):
        self.path(pod_id)
        result = self.call("POST", "/graphql", {"query": query, "variables": {"id": pod_id}})
        data = result.get("data")
        if result.get("errors") or not isinstance(data, dict) or field not in data:
            raise RunpodError("Runpod GraphQL request failed; response body suppressed")
        return data[field]

    def get(self, pod_id):
        pod = self.graphql("query($id:String!){pod(input:{podId:$id}){id name desiredStatus costPerHr "
            "imageName gpuCount machine{secureCloud}}}", pod_id, "pod")
        if not isinstance(pod, dict):
            raise RunpodError("Runpod Pod details are unavailable")
        return pod

    def stop(self, pod_id):
        self.graphql("mutation($id:String!){podStop(input:{podId:$id}){id desiredStatus}}", pod_id, "podStop")

    def terminate(self, pod_id):
        self.graphql("mutation($id:String!){podTerminate(input:{podId:$id})}", pod_id, "podTerminate")


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
