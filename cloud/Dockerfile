# GPU image for the Gemma-3 GPS pipeline (built for the 27B, works for any size).
#
# The default PyPI torch wheels are CUDA-enabled, so `uv sync` on a CUDA base
# gives a GPU build automatically. The macOS-only mlx-lm dependency is guarded
# by a platform marker in pyproject.toml and is skipped on Linux.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY . /app

# Resolve the environment (skip dev tools). Torch pulls its CUDA wheel here.
RUN uv sync --no-dev

# HF_TOKEN is provided at run time (Gemma is a gated model — accept the license
# on huggingface.co first). PYTORCH_ENABLE_MPS_FALLBACK is harmless on CUDA.
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Default: run the full 27B pipeline. Override the folder as the CMD arg.
CMD ["bash", "cloud/run_all.sh", "gemma3_27b"]
