"""Resolves the checkpoint/gallery paths the API service uses at startup.

Defaults match `writer_identification.backend.cli`'s own defaults (relative
to the repo root), but are overridable via `WRITER_ID_CHECKPOINT_PATH` /
`WRITER_ID_GALLERY_PATH` env vars -- this is what makes the Docker
volume-mount scenario (Part 4) work without any code change: the container
sets those env vars to point at the mounted volume.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from writer_identification.backend.cli import DEFAULT_CHECKPOINT, DEFAULT_GALLERY


@dataclass(frozen=True)
class ServiceConfig:
    """Filesystem locations the API service loads at startup."""

    checkpoint_path: Path
    gallery_path: Path


def load_service_config() -> ServiceConfig:
    checkpoint_path = Path(os.environ.get("WRITER_ID_CHECKPOINT_PATH", DEFAULT_CHECKPOINT))
    gallery_path = Path(os.environ.get("WRITER_ID_GALLERY_PATH", DEFAULT_GALLERY))
    return ServiceConfig(checkpoint_path=checkpoint_path, gallery_path=gallery_path)
