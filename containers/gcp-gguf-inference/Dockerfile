# syntax=docker/dockerfile:1.7
# Build with the curated Hugging Face release directory as the Docker context.
# The exact model, projector, template, CUDA bases, and llama.cpp source are pinned.
ARG LLAMA_CPP_COMMIT=0faee5004297c3bcfa40b7bf11750127b8c1fd7d
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04@sha256:50efab398f76258daa91ceebb33b6467e40217c67ea44fb5a2cebc6be7d9cce3 AS llama-build

ARG LLAMA_CPP_COMMIT
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates cmake g++ git libgomp1 ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch b11384 https://github.com/ggml-org/llama.cpp.git llama.cpp \
    && test "$(git -C llama.cpp rev-parse HEAD)" = "$LLAMA_CPP_COMMIT"

WORKDIR /src/llama.cpp
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
    && cmake --build build --config Release --target llama-server -j 8 \
    && mkdir -p /out/bin \
    && cp -a build/bin/llama-server /out/bin/ \
    && find build -name '*.so*' -exec cp -P {} /out/bin/ \;

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:23debbe74125dc84df96df79cff42079b3b15265c27140714fd27b5aa718faa4

ARG LLAMA_CPP_COMMIT
ARG MODEL_HF_REVISION
ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/opt/llama.cpp/bin:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama-build /out/bin /opt/llama.cpp/bin
RUN /opt/llama.cpp/bin/llama-server --help 2>&1 | grep -q -- '--chat-template-file' \
    && test "$MODEL_HF_REVISION" = "477e3caf32eced2a9e002ead1edccc9a5c458f0c"
COPY model-Q4_K_M.gguf mmproj-BF16.gguf chat_template.jinja /model/
RUN printf '%s  %s\n' \
        bd2ec1357e84b27b58ec0b1c83e143028f9b54e42989dc9f3db7c59aed870639 /model/model-Q4_K_M.gguf \
        3248a0391bed86eef3842121dab70d4657da244bde47ae02ce37580351bd0138 /model/mmproj-BF16.gguf \
        4813f6c608147061ba44f17b01aa344de7b5036cb97bb39200a3b2bb6e26a0c5 /model/chat_template.jinja \
    | sha256sum --check --status

LABEL org.opencontainers.image.title="EduGraph Classifier Qwen3.8-27B GGUF" \
      org.opencontainers.image.source="https://github.com/christian-bick/edugraph-classify" \
      org.opencontainers.image.description="Pinned llama.cpp server for the verified EduGraph Q4_K_M candidate" \
      org.opencontainers.image.model.revision="$MODEL_HF_REVISION" \
      org.opencontainers.image.llama.cpp.commit="$LLAMA_CPP_COMMIT"

EXPOSE 8080
ENTRYPOINT ["/opt/llama.cpp/bin/llama-server"]
CMD ["-m", "/model/model-Q4_K_M.gguf", "--mmproj", "/model/mmproj-BF16.gguf", "--jinja", "--chat-template-file", "/model/chat_template.jinja", "--ctx-size", "2048", "--ubatch-size", "1024", "--parallel", "1", "--n-gpu-layers", "all", "--fit", "off", "--host", "0.0.0.0", "--port", "8080"]
