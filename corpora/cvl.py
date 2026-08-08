"""CVL Database loader (writer identification / verification).

The CVL Database is not distributed on the HuggingFace Hub (Creative Commons
BY-NC 3.0, obtained directly from TU Wien: https://doi.org/10.5281/zenodo.1492267).
This loader instead expects the raw archive to already be converted into
HuggingFace's `imagefolder` convention under `config.root_dir`: one
subdirectory per split (e.g. `train/`, `test/`) containing the writer's
handwritten-page crops plus a `metadata.jsonl` with a `file_name` and
`writer_id` column per row. That conversion is a one-off manual/DVC-pipeline
step outside this engine's scope, since it depends on exactly which of CVL's
distributed archive layouts was downloaded.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from datasets import DatasetDict, load_dataset

from corpora._common import standardize_dataset_dict
from corpora.config import CVLConfig
from corpora.schema import DatasetSplit, WriterIdentityMeta


def standardize_cvl_row(row: Mapping[str, Any], split: DatasetSplit) -> dict[str, Any]:
    """Map one `imagefolder`-ingested CVL row onto the standardized writer-identity schema."""
    meta = WriterIdentityMeta(writer_id=str(row["writer_id"]), split=split)
    return {"image": row["image"], **meta.model_dump(mode="json")}


def load_cvl(config: CVLConfig, raw: DatasetDict | None = None) -> DatasetDict:
    """Load the CVL writer-identification/verification dataset, standardized to
    `WriterIdentityMeta` + `image` columns.

    `raw` allows injecting an already-loaded `DatasetDict` (used by tests
    against a tiny fixture directory); when omitted, `config.root_dir` is read
    via HuggingFace's `imagefolder` loader.
    """
    if raw is None:
        raw = load_dataset("imagefolder", data_dir=str(config.root_dir))
    return standardize_dataset_dict(raw, standardize_cvl_row)
