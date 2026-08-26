"""Configuration for the writer-identification adapter."""

from __future__ import annotations

from pathlib import Path

from handwriting_engine.embeddings.config import EmbeddingConfig
from handwriting_engine.preprocessing.config import PreprocessingConfig
from pydantic import BaseModel, ConfigDict


class IdentificationAppConfig(BaseModel):
    """Configuration for `IdentificationAdapter`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preprocessing: PreprocessingConfig | None = None
    """Preprocessing pipeline to run before embedding a query/gallery image.
    `None` (the default) skips the engine's preprocessing pipeline entirely
    -- raw grayscale straight to the backbone, matching the distribution
    every training run and evaluation script (`scripts/evaluate_writer_id.py`,
    `scripts/probe_corpus_separability.py`) actually uses. Passing an
    explicit `PreprocessingConfig` runs perspective correction/deskew/
    denoise/CLAHE/binarization/morphology first -- untested against this
    project's trained checkpoints (a spot check of `scripts/build_gallery.py
    spot_check=true` under full preprocessing scored 3/8 correct, vs. ~62%
    top-1 without it -- do not enable without re-validating accuracy)."""
    embedding: EmbeddingConfig = EmbeddingConfig()
    checkpoint_path: Path | None = None
    """Path to this downstream project's trained checkpoint (e.g.
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
