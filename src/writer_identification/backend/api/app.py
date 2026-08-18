"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from writer_identification.backend.api.routes import router
from writer_identification.backend.api.state import AppState, build_app_state


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the identification state once at startup, unless a caller
    (tests) has already preset `app.state.identification` before the
    lifespan ran."""
    if not hasattr(app.state, "identification"):
        app.state.identification = build_app_state()
    yield


def create_app(state: AppState | None = None) -> FastAPI:
    """Build the FastAPI application.

    `state` allows presetting the identification state (used by tests, to
    inject a tiny model instead of loading the real checkpoint+gallery);
    when omitted, the real state is built at startup via `lifespan`.
    """
    app = FastAPI(title="Writer Identification", lifespan=_lifespan)
    app.include_router(router)
    if state is not None:
        app.state.identification = state
    return app


app = create_app()
