"""Pydantic request/response schemas for the API service."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Response for the liveness/readiness check."""

    model_config = ConfigDict(frozen=True)

    status: str = "ok"


class MatchSchema(BaseModel):
    """One ranked gallery candidate for a query image."""

    model_config = ConfigDict(frozen=True)

    label: str
    similarity: float


class IdentifyResponse(BaseModel):
    """Response for the `/identify` endpoint."""

    model_config = ConfigDict(frozen=True)

    matches: list[MatchSchema]


class EnrollResponse(BaseModel):
    """Response for the `/enroll` endpoint."""

    model_config = ConfigDict(frozen=True)

    writer_id: str
    samples_added: int
    gallery_size: int
