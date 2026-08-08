"""Unit tests for corpora.iam.

Exercises the full `load_iam` path against an in-memory `DatasetDict` shaped
like the real `Teklia/IAM-line` schema (`image`, `text` columns), so no
network access is needed.
"""

from __future__ import annotations

from collections.abc import Callable

from datasets import Dataset, DatasetDict
from PIL import Image

from corpora.config import IAMConfig
from corpora.iam import load_iam


def test_load_iam_standardizes_rows_without_network(
    make_tiny_image: Callable[[], Image.Image],
) -> None:
    raw = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {"image": make_tiny_image(), "text": "hello world"},
                    {"image": make_tiny_image(), "text": "second line"},
                ]
            ),
            "test": Dataset.from_list([{"image": make_tiny_image(), "text": "test row"}]),
        }
    )

    result = load_iam(IAMConfig(), raw=raw)

    assert set(result.keys()) == {"train", "test"}
    assert result["train"]["transcription"] == ["hello world", "second line"]
    assert result["train"]["writer_id"] == [None, None]
    assert result["train"]["split"] == ["train", "train"]
    assert result["test"]["transcription"] == ["test row"]
    assert result["test"]["split"] == ["test"]
    assert isinstance(result["train"][0]["image"], Image.Image)
