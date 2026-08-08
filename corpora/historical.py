"""IAM-HistDB loader (historical handwriting recognition).

IAM-HistDB (George Washington, Parzival, Ratsprotokolle/READ2016, and related
sub-collections) is not distributed on the HuggingFace Hub; it is obtained
directly from the University of Fribourg's HisDoc project. As with `cvl.py`,
this loader expects the raw archive already converted into HuggingFace's
`imagefolder` convention under `config.root_dir`: one subdirectory per split,
each with a `metadata.jsonl` carrying `file_name`, `collection` (which
sub-collection the page/line belongs to), and an optional `transcription`
(omitted for sub-collections that only ship page images without line-aligned
ground truth). That conversion is a one-off manual/DVC-pipeline step outside
this engine's scope, since it depends on which sub-collection(s) were
downloaded and their native ground-truth format.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from datasets import DatasetDict, load_dataset

from corpora._common import standardize_dataset_dict
from corpora.config import IAMHistDBConfig
from corpora.schema import DatasetSplit, HistoricalMeta


def standardize_historical_row(row: Mapping[str, Any], split: DatasetSplit) -> dict[str, Any]:
    """Map one `imagefolder`-ingested IAM-HistDB row onto the standardized historical schema."""
    meta = HistoricalMeta(
        transcription=row.get("transcription"), collection=row["collection"], split=split
    )
    return {"image": row["image"], **meta.model_dump(mode="json")}


def load_iam_histdb(config: IAMHistDBConfig, raw: DatasetDict | None = None) -> DatasetDict:
    """Load the IAM-HistDB historical handwriting dataset, standardized to
    `HistoricalMeta` + `image` columns.

    `raw` allows injecting an already-loaded `DatasetDict` (used by tests
    against a tiny fixture directory); when omitted, `config.root_dir` is read
    via HuggingFace's `imagefolder` loader.
    """
    if raw is None:
        raw = load_dataset("imagefolder", data_dir=str(config.root_dir))
    return standardize_dataset_dict(raw, standardize_historical_row)
