"""API route handlers: identify a query image against the gallery, enroll
new writers, health check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from handwriting_engine.api.imaging import decode_image
from numpy.typing import NDArray

from writer_identification.backend.api.schemas import (
    EnrollResponse,
    HealthResponse,
    IdentifyResponse,
    MatchSchema,
)
from writer_identification.backend.api.state import AppState

router = APIRouter()

_INDEX_HTML = (Path(__file__).resolve().parent / "static" / "index.html").read_text()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """Minimal browser frontend for `/enroll` and `/identify` -- read once
    at import time rather than per-request since the file never changes at
    runtime."""
    return _INDEX_HTML


def _app_state(request: Request) -> AppState:
    state: AppState = request.app.state.identification
    return state


async def _decode_uploaded_image(file: UploadFile) -> NDArray[np.uint8]:
    data = await file.read()
    try:
        return decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness check."""
    return HealthResponse()


@router.post("/identify", response_model=IdentifyResponse)
async def identify(request: Request, file: UploadFile, top_k: int = 5) -> IdentifyResponse:
    """Rank an uploaded handwriting sample against the configured gallery.

    Expects an already-cropped single word/line handwriting sample -- no
    perspective correction, denoising, or segmentation happens anywhere in
    this pipeline. A full page photo will be embedded as if it were one
    giant "word," producing a meaningless result, not just a less accurate
    one.
    """
    state = _app_state(request)
    image = await _decode_uploaded_image(file)

    try:
        matches = state.adapter.identify(image, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IdentifyResponse(
        matches=[MatchSchema(label=match.label, similarity=match.similarity) for match in matches]
    )


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(
    request: Request,
    writer_id: str = Form(...),
    files: list[UploadFile] = File(...),  # noqa: B008 -- idiomatic FastAPI dependency default
) -> EnrollResponse:
    """Enroll one or more samples for `writer_id` into the gallery, persisting
    the updated gallery to disk immediately.

    Same input contract as `/identify`: already-cropped single word/line
    handwriting samples, not full page photos.
    """
    state = _app_state(request)
    images = [await _decode_uploaded_image(file) for file in files]

    async with state.enroll_lock:
        for image in images:
            state.adapter.enroll(image, writer_id)
        state.adapter.save_gallery(state.gallery_path)
        gallery_size = state.adapter.gallery_size

    return EnrollResponse(
        writer_id=writer_id, samples_added=len(images), gallery_size=gallery_size
    )
