"""Configuration for the writer-identification adapter."""

from __future__ import annotations

from pathlib import Path

from handwriting_engine.embeddings.config import EmbeddingConfig
from handwriting_engine.preprocessing.config import PreprocessingConfig
from pydantic import BaseModel, ConfigDict


class IdentificationAppConfig(BaseModel):
    """Configuration for `IdentificationAdapter`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    checkpoint_path: Path | None = None
    """Path to this child project's trained checkpoint (e.g.
    `models/checkpoints/writer_id/best_model.pt`, written by
    `scripts/train.py corpus=writer_id`). Loaded into a freshly constructed
    `EmbeddingModel` when the adapter builds its own model (i.e. no
    `embedding_model` is passed to `IdentificationAdapter` directly -- that
    still takes priority, e.g. for tests injecting a tiny model). `None` (the
    default) builds an untrained model."""
    gallery_path: Path | None = None
    """Path to a `.npz` gallery of known embeddings to rank against --
    `embeddings` (num_samples x embedding_dim, L2-normalized) and `labels`
    (num_samples, string identity labels) arrays, same convention
    `scripts/inference.py --gallery` uses. Required for `.identify(...)`
    unless a gallery is passed directly to `IdentificationAdapter`; without
    either, the adapter can still `.embed(...)` a single image."""
