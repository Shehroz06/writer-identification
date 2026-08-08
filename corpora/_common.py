"""Shared helpers for standardizing a raw HuggingFace `DatasetDict` -- however
its splits were produced (Hub download or local `imagefolder` ingestion) --
onto this project's per-task row schema. Reused by every loader in this
module per the "no duplicated logic" rule."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from datasets import Dataset, DatasetDict

from corpora.schema import DatasetSplit

_SPLIT_ALIASES = {
    "train": DatasetSplit.TRAIN,
    "validation": DatasetSplit.VALIDATION,
    "val": DatasetSplit.VALIDATION,
    "test": DatasetSplit.TEST,
}


def resolve_split(split_name: str) -> DatasetSplit:
    """Map a raw split name (a Hub split key or an `imagefolder` subdirectory
    name) onto this project's canonical `DatasetSplit`.

    Raises `ValueError` on an unrecognized split name rather than silently
    guessing, since a misattributed split would silently corrupt train/test
    separation for every downstream consumer.
    """
    try:
        return _SPLIT_ALIASES[split_name]
    except KeyError as exc:
        raise ValueError(
            f"Unrecognized dataset split name {split_name!r}; expected one of "
            f"{sorted(_SPLIT_ALIASES)}."
        ) from exc


def standardize_dataset_dict(
    raw: DatasetDict,
    row_fn: Callable[[Mapping[str, Any], DatasetSplit], dict[str, Any]],
) -> DatasetDict:
    """Apply `row_fn` across every split of `raw`, replacing each split's raw
    columns with the standardized columns `row_fn` returns (`row_fn` is
    responsible for passing through the `image` column unchanged).
    """
    standardized: dict[str, Dataset] = {}
    for split_name, split_dataset in raw.items():
        split = resolve_split(split_name)
        standardized[split_name] = split_dataset.map(
            lambda row, split=split: row_fn(row, split),
            remove_columns=split_dataset.column_names,
        )
    return DatasetDict(standardized)
