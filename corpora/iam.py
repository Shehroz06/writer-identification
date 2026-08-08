"""IAM Handwriting Database loader.

Source: `Teklia/IAM-line` on the HuggingFace Hub -- line-level crops of the IAM
Handwriting Database with two raw columns: `image` (a decoded PIL image) and
`text` (the transcription). IAM does not carry writer identifiers at this
granularity, so `writer_id` is always `None` here; it also provides
paragraph/form-level ground truth (this project's "paragraphs" task) and the
word/line bounding boxes used for the "segmentation" task, which are exposed
by other Hub configs of the same underlying database rather than duplicated
here. License: MIT (for this Hub mirror); non-commercial research use for the
original IAM data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from datasets import DatasetDict, load_dataset

from corpora._common import standardize_dataset_dict
from corpora.config import IAMConfig
from corpora.schema import DatasetSplit, RecognitionMeta


def standardize_iam_row(row: Mapping[str, Any], split: DatasetSplit) -> dict[str, Any]:
    """Map one raw `Teklia/IAM-line` row onto the standardized recognition schema."""
    meta = RecognitionMeta(transcription=row["text"], writer_id=None, split=split)
    return {"image": row["image"], **meta.model_dump(mode="json")}


def load_iam(config: IAMConfig, raw: DatasetDict | None = None) -> DatasetDict:
    """Load the IAM line-level recognition dataset, standardized to
    `RecognitionMeta` + `image` columns.

    `raw` allows injecting an already-loaded `DatasetDict` (used by tests to
    exercise standardization without a network call); when omitted, the Hub
    dataset is downloaded/cached under `config.cache_dir`.
    """
    if raw is None:
        raw = load_dataset(config.hub_id, cache_dir=str(config.cache_dir))
    return standardize_dataset_dict(raw, standardize_iam_row)
