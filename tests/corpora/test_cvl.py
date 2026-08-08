"""Unit tests for corpora.cvl.

Builds a tiny on-disk directory following HuggingFace's `imagefolder`
convention (per-split subdirectories + metadata.jsonl) and exercises
`load_cvl` end-to-end against it -- entirely local, no network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from corpora.config import CVLConfig
from corpora.cvl import load_cvl


def _write_imagefolder_split(split_dir: Path, rows: list[dict[str, str]]) -> None:
    split_dir.mkdir(parents=True)
    with (split_dir / "metadata.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    for row in rows:
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(split_dir / row["file_name"])


def test_load_cvl_standardizes_writer_identity_rows(tmp_path: Path) -> None:
    root_dir = tmp_path / "cvl"
    _write_imagefolder_split(
        root_dir / "train",
        [
            {"file_name": "a.png", "writer_id": "writer-1"},
            {"file_name": "b.png", "writer_id": "writer-2"},
        ],
    )
    _write_imagefolder_split(
        root_dir / "test",
        [{"file_name": "c.png", "writer_id": "writer-3"}],
    )

    result = load_cvl(CVLConfig(root_dir=root_dir))

    assert set(result.keys()) == {"train", "test"}
    assert sorted(result["train"]["writer_id"]) == ["writer-1", "writer-2"]
    assert result["train"]["split"] == ["train", "train"]
    assert result["test"]["writer_id"] == ["writer-3"]
    assert result["test"]["split"] == ["test"]
    assert isinstance(result["train"][0]["image"], Image.Image)
