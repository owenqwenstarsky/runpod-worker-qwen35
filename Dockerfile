ARG LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
FROM ${LLAMA_CPP_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface \
    MODEL_REPO=TeichAI/Qwen3.5-4B-Claude-Opus-Reasoning-GGUF \
    MODEL_PATH=/models/model.gguf \
    RUNPOD_MODEL_NAME=qwen3.5-4b-claude-opus-reasoning-q8_0 \
    LLAMA_SERVER_BIN=/app/llama-server \
    LLAMA_SERVER_HOST=127.0.0.1 \
    LLAMA_SERVER_PORT=8080 \
    LLAMA_CTX_SIZE=8192 \
    LLAMA_PARALLEL=1 \
    LLAMA_GPU_LAYERS=999 \
    LLAMA_BATCH=1024 \
    LLAMA_JINJA=1

WORKDIR /worker

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install "huggingface_hub[hf_xet]"

RUN mkdir -p /models

RUN python3 -m huggingface_hub download "${MODEL_REPO}" \
        --local-dir /models \
        --include "*.q8_0.gguf" \
        --include "*.Q8_0.gguf" \
        --exclude "*mmproj*" && \
    MODEL_FILE="$(find /models -type f \( -iname '*.q8_0.gguf' -o -iname '*.Q8_0.gguf' \) ! -iname '*mmproj*' | head -n 1)" && \
    test -n "${MODEL_FILE}" && \
    if [ "${MODEL_FILE}" != "${MODEL_PATH}" ]; then mv "${MODEL_FILE}" "${MODEL_PATH}"; fi && \
    find /models -mindepth 1 ! -path "${MODEL_PATH}" -exec rm -rf {} + && \
    rm -rf "${HF_HOME}"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
