# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:23debbe74125dc84df96df79cff42079b3b15265c27140714fd27b5aa718faa4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/tmp/huggingface

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libgomp1 \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/uv \
    && /opt/uv/bin/pip install --no-cache-dir uv==0.7.12

ENV PATH="/opt/uv/bin:$PATH" \
    UV_NO_CACHE=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group training --no-install-project --python /usr/bin/python3.12

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --group training --python /usr/bin/python3.12

ENTRYPOINT ["/app/.venv/bin/python", "-m", "edugraph_classify.vlm_trainer"]
