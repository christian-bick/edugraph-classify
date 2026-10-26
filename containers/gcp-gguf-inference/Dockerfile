# syntax=docker/dockerfile:1.7
# Template rendered only from a strictly validated, reviewed JSON release profile.
# Copy the proven SM89 runtime without inheriting its previous model-weight layer.
FROM @@RUNTIME_IMAGE@@ AS verified-runtime

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:23debbe74125dc84df96df79cff42079b3b15265c27140714fd27b5aa718faa4

ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/opt/llama.cpp/bin:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=verified-runtime /opt/llama.cpp/bin /opt/llama.cpp/bin
RUN /opt/llama.cpp/bin/llama-server --version > /tmp/llama-version.txt 2>&1 \
    && cat /tmp/llama-version.txt \
    && grep -F -- '@@LLAMA_CPP_SHORT_COMMIT@@' /tmp/llama-version.txt \
    && /opt/llama.cpp/bin/llama-server --help > /tmp/llama-help.txt 2>&1 \
    && for flag in --chat-template-file --ctx-size --batch-size --ubatch-size --mmproj --fit; do \
         grep -Fq -- "$flag" /tmp/llama-help.txt || exit 1; \
       done \
    && rm /tmp/llama-version.txt /tmp/llama-help.txt

COPY model-Q4_K_M.gguf mmproj-BF16.gguf chat_template.jinja /model/
RUN printf '%s  %s\n' \
        @@MODEL_SHA256@@ /model/model-Q4_K_M.gguf \
        @@PROJECTOR_SHA256@@ /model/mmproj-BF16.gguf \
        @@TEMPLATE_SHA256@@ /model/chat_template.jinja \
    | sha256sum --check --status

LABEL org.opencontainers.image.title="EduGraph Classifier Qwen3.8-27B GGUF" \
      org.opencontainers.image.source="https://github.com/christian-bick/edugraph-classify" \
      org.opencontainers.image.description="Pinned llama.cpp server for the verified EduGraph Q4_K_M candidate" \
      org.opencontainers.image.model.revision="@@MODEL_HF_REVISION@@" \
      org.opencontainers.image.llama.cpp.commit="@@LLAMA_CPP_COMMIT@@" \
      org.opencontainers.image.runtime.source="@@RUNTIME_IMAGE@@"

EXPOSE 8080
ENTRYPOINT ["/opt/llama.cpp/bin/llama-server"]
CMD ["-m", "/model/model-Q4_K_M.gguf", "--mmproj", "/model/mmproj-BF16.gguf", "--jinja", "--chat-template-file", "/model/chat_template.jinja", "--ctx-size", "@@CONTEXT_TOKENS@@", "--batch-size", "@@BATCH_SIZE@@", "--ubatch-size", "@@UBATCH_SIZE@@", "--parallel", "1", "--n-gpu-layers", "all", "--fit", "off", "--host", "0.0.0.0", "--port", "8080"]
