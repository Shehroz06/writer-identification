"""Standardized row schemas for every dataset task this module loads.

Each loader (`iam`, `cvl`, `signatures`, `historical`) maps a source-specific raw
row onto one of these Pydantic models before handing it back as a plain dict for
a HuggingFace `Dataset`/`DatasetDict`. Images are deliberately excluded from these
models and passed through untouched as the dataset's native `image` feature
(PIL-decodable) -- Pydantic validation only covers the non-image metadata.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DatasetSplit(StrEnum):
    """Canonical split names every loader normalizes its source's split
    naming onto (e.g. a source's "val" folder becomes VALIDATION)."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class RowMeta(BaseModel):
    """Base class for a single standardized row's non-image metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: DatasetSplit


class RecognitionMeta(RowMeta):
    """Row metadata for offline handwriting recognition (and, since IAM's forms
    carry the same ground truth at multiple granularities, paragraph-level and
    segmentation-region tasks reuse this schema too)."""

    transcription: str
    writer_id: str | None = None


class WriterIdentityMeta(RowMeta):
    """Row metadata for writer identification / verification: every sample is
    tied to a writer, with no other label required."""

    writer_id: str


class SignatureMeta(RowMeta):
    """Row metadata for offline signature verification: a signature image
    attributed to a claimed writer, labeled genuine or a skilled forgery."""

    writer_id: str
    is_genuine: bool


class HistoricalMeta(RowMeta):
    """Row metadata for historical handwriting recognition. `transcription` is
    optional since some historical collections only ship page-level images
    without line-aligned ground truth."""

    transcription: str | None = None
    collection: str
