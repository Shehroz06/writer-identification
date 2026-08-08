"""Unit tests for corpora.historical.

Builds a tiny on-disk directory following HuggingFace's `imagefolder`
convention and exercises `load_iam_histdb` end-to-end -- entirely local, no
network access. Covers both the with-transcription and
without-transcription (page-only) cases described in the loader's docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from corpora.config import IAMHistDBConfig
from corpora.historical import load_iam_histdb


def test_load_iam_histdb_standardizes_rows_with_and_without_transcription(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "iam_histdb"
    train_dir = root_dir / "train"
    train_dir.mkdir(parents=True)
    rows = [
        {"file_name": "a.png", "collection": "parzival", "transcription": "et dixit"},
        {"file_name": "b.png", "collection": "washington"},
    ]
    with (train_dir / "metadata.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    for row in rows:
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(train_dir / row["file_name"])

    result = load_iam_histdb(IAMHistDBConfig(root_dir=root_dir))

    standardized_rows = list(result["train"])
    by_collection = {row["collection"]: row for row in standardized_rows}

    assert by_collection["parzival"]["transcription"] == "et dixit"
    assert by_collection["washington"]["transcription"] is None
    assert all(row["split"] == "train" for row in standardized_rows)
