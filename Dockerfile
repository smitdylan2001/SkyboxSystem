# CUDA 13.0 runtime needs an NVIDIA driver >= 580 on the host (check `nvidia-smi`).
# For an older driver, build with: --build-arg BASE_IMAGE=pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime
ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODELS_DIR=/models \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    DATA_DIR=/data \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep the base image's torch/torchvision (never let pip swap them for another CUDA build).
COPY requirements.txt .
RUN pip freeze | grep -iE '^(torch|torchvision)==' > /tmp/constraints.txt \
 && pip install -c /tmp/constraints.txt -r requirements.txt \
 && pip install --no-deps "moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"

COPY app ./app
COPY tools ./tools

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# One process: the API and the single GPU worker thread share the job queue.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--workers", "1"]
