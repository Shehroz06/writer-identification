"""Application state: the identification adapter, built once at service
startup and reused across requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from writer_identification.backend.adapter import IdentificationAdapter
from writer_identification.backend.api.config import load_service_config
from writer_identification.backend.config import IdentificationAppConfig


@dataclass(frozen=True)
class AppState:
    """Everything a request handler needs, built once at startup."""

    adapter: IdentificationAdapter
    gallery_path: Path
    """Where `/enroll` persists the gallery after mutating it -- kept
    separate from the adapter itself since an adapter can be built without
    ever having been configured with a path (e.g. tests injecting a gallery
    directly)."""
    enroll_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    """Guards `/enroll`'s gallery mutation + save -- without it, two
    concurrent enrollments could race on the in-memory array mutation or
    clobber each other's `np.savez` write."""


def build_app_state(
    adapter: IdentificationAdapter | None = None, gallery_path: Path | None = None
) -> AppState:
    """Load configuration and construct the adapter.

    `adapter`/`gallery_path` allow injecting an already-built adapter and
    its save target (used by tests, to avoid downloading/loading the real
    checkpoint+gallery); when omitted, a real `IdentificationAdapter` is
    constructed from the resolved checkpoint/gallery paths, failing fast
    (before the server starts accepting traffic) if either path is missing
    -- a missing volume mount should crash the container immediately, not
    surface as a 500 on the first request.
    """
    if adapter is not None:
        if gallery_path is None:
            raise ValueError("gallery_path is required when injecting an adapter")
        return AppState(adapter=adapter, gallery_path=gallery_path)

    service_config = load_service_config()
    if not service_config.checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {service_config.checkpoint_path} -- set "
            "WRITER_ID_CHECKPOINT_PATH or mount the checkpoint volume."
        )
    if not service_config.gallery_path.exists():
        raise FileNotFoundError(
            f"No gallery at {service_config.gallery_path} -- set "
            "WRITER_ID_GALLERY_PATH or mount the gallery volume."
        )

    config = IdentificationAppConfig(
        checkpoint_path=service_config.checkpoint_path,
        gallery_path=service_config.gallery_path,
    )
    return AppState(
        adapter=IdentificationAdapter.from_config(config),
        gallery_path=service_config.gallery_path,
    )
