# syntax=docker/dockerfile:1.7
# Build llama.cpp from a reviewed release commit. The image contains no model weights.
ARG LLAMA_CPP_COMMIT=0faee5004297c3bcfa40b7bf11750127b8c1fd7d
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04@sha256:50efab398f76258daa91ceebb33b6467e40217c67ea44fb5a2cebc6be7d9cce3 AS llama-build

ARG LLAMA_CPP_COMMIT
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates cmake g++ git libgomp1 ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch b11384 https://github.com/ggml-org/llama.cpp.git llama.cpp \
    && test "$(git -C llama.cpp rev-parse HEAD)" = "$LLAMA_CPP_COMMIT"

WORKDIR /src/llama.cpp
# A40, L4/L40S, and H100. A changed GPU family needs a reviewed image rebuild.
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES='86;89;90' \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
    && cmake --build build --config Release --target llama-quantize llama-server -j "$(nproc)" \
    && mkdir -p /out/bin \
    && cp -a build/bin/llama-quantize build/bin/llama-server /out/bin/ \
    && find build -name '*.so*' -exec cp -P {} /out/bin/ \;

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:23debbe74125dc84df96df79cff42079b3b15265c27140714fd27b5aa718faa4

ARG LLAMA_CPP_COMMIT
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/tmp/huggingface \
    PYTHONPATH=/opt/llama.cpp:/opt/llama.cpp/gguf-py \
    LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates git libgomp1 python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/uv \
    && /opt/uv/bin/pip install --no-cache-dir uv==0.7.12

ENV PATH="/opt/uv/bin:/opt/llama.cpp/build/bin:$PATH" \
    UV_NO_CACHE=1 \
    EDUGRAPH_LLAMA_CPP_COMMIT=$LLAMA_CPP_COMMIT

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group training --group inference-benchmark --no-install-project --python /usr/bin/python3.12

COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --group training --group inference-benchmark --python /usr/bin/python3.12

COPY --from=llama-build /src/llama.cpp/convert_hf_to_gguf.py /opt/llama.cpp/
COPY --from=llama-build /src/llama.cpp/.git /opt/llama.cpp/.git
COPY --from=llama-build /src/llama.cpp/LICENSE /opt/llama.cpp/LICENSE
COPY --from=llama-build /src/llama.cpp/requirements.txt /opt/llama.cpp/requirements.txt
COPY --from=llama-build /src/llama.cpp/requirements /opt/llama.cpp/requirements
COPY --from=llama-build /src/llama.cpp/conversion /opt/llama.cpp/conversion
COPY --from=llama-build /src/llama.cpp/gguf-py /opt/llama.cpp/gguf-py
COPY --from=llama-build /out/bin /opt/llama.cpp/build/bin

ARG CODE_COMMIT=uncommitted
ENV EDUGRAPH_CODE_COMMIT=$CODE_COMMIT

# CPU-only build checks: import the worker and converter, and inspect required CLI flags.
RUN --network=none uv run --no-sync python -c \
    "import torch, transformers, peft, sentencepiece, gguf; from edugraph_classify import inference_benchmark_worker; assert callable(inference_benchmark_worker.main); assert torch.version.cuda == '12.6', torch.version.cuda" \
    && test "$(git -C /opt/llama.cpp rev-parse HEAD)" = "$EDUGRAPH_LLAMA_CPP_COMMIT" \
    && uv run --no-sync python /opt/llama.cpp/convert_hf_to_gguf.py --help | grep -q -- '--mmproj' \
    && uv run --no-sync python /opt/llama.cpp/convert_hf_to_gguf.py --help | grep -q -- '--no-mtp' \
    && /opt/llama.cpp/build/bin/llama-quantize --help 2>&1 | grep -q -- 'Q4_K_M' \
    && /opt/llama.cpp/build/bin/llama-server --help 2>&1 | grep -q -- '--chat-template-file' \
    && uv run --no-sync edugraph-classify inference-benchmark --help

ENTRYPOINT ["/app/.venv/bin/python", "-m", "edugraph_classify.inference_benchmark_worker"]
