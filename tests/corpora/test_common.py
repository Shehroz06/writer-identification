"""Unit tests for corpora._common."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from datasets import Dataset, DatasetDict

from corpora._common import resolve_split, standardize_dataset_dict
from corpora.schema import DatasetSplit


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("train", DatasetSplit.TRAIN),
        ("validation", DatasetSplit.VALIDATION),
        ("val", DatasetSplit.VALIDATION),
        ("test", DatasetSplit.TEST),
    ],
)
def test_resolve_split_maps_known_aliases(raw_name: str, expected: DatasetSplit) -> None:
    assert resolve_split(raw_name) is expected


def test_resolve_split_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unrecognized dataset split name"):
        resolve_split("unknown_split")


def test_standardize_dataset_dict_applies_row_fn_per_split() -> None:
    raw = DatasetDict(
        {
            "train": Dataset.from_list([{"value": 1}, {"value": 2}]),
            "test": Dataset.from_list([{"value": 3}]),
        }
    )

    def row_fn(row: Mapping[str, Any], split: DatasetSplit) -> dict[str, Any]:
        return {"value": row["value"] * 10, "split": split.value}

    standardized = standardize_dataset_dict(raw, row_fn)

    assert set(standardized.keys()) == {"train", "test"}
    assert standardized["train"]["value"] == [10, 20]
    assert standardized["train"]["split"] == ["train", "train"]
    assert standardized["test"]["value"] == [30]
    assert standardized["test"]["split"] == ["test"]
