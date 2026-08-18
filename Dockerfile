# syntax=docker/dockerfile:1
#
# Writer Identification API service. CPU-only, uv-managed, same base image
# and opencv runtime libs as the handwriting-engine's own Dockerfile.
#
# The engine dependency is a private GitHub repo (see pyproject.toml) --
# BuildKit's SSH-forwarding (--mount=type=ssh) lets `uv sync` clone it using
# the host's existing SSH key for the duration of each RUN step, without
# ever writing the key into an image layer. Build with:
#     DOCKER_BUILDKIT=1 docker build --ssh default -t writer-identification-api .

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 openssh-client git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/
WORKDIR /app

RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Install dependencies first (cached across builds unless pyproject.toml/
# uv.lock change) before copying source, so editing application code doesn't
# invalidate the dependency-install layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=ssh uv sync --frozen --no-install-project --no-dev

# README.md is required at build time: pyproject.toml's hatchling backend
# reads it for the package's `readme` field.
COPY README.md ./
COPY src/ src/
RUN --mount=type=ssh uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
# HuggingFace Hub cache for the DINOv2 checkpoint, persisted via a mounted
# volume in production so it survives container restarts instead of
# re-downloading on every cold start.
ENV HF_HOME="/app/.cache/huggingface"

# Checkpoint + gallery are volume-mounted, not baked in -- see README.
ENV WRITER_ID_CHECKPOINT_PATH="/models/writer_id/best_model.pt"
ENV WRITER_ID_GALLERY_PATH="/models/writer_id/gallery.npz"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "writer_identification.backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
