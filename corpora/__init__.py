"""Dataset loading glue for this project's corpora (CVL + Firemaker/CERUG for
the actual training pipeline; IAM + IAM-HistDB kept as reference loaders not
currently used by any script here).

Lives outside `writer_identification` (the installed package) deliberately:
these are concrete, named corpus loaders with task-specific schema baked in,
not generic engine behavior. Consumed by `scripts/`, not by
`handwriting_engine` itself."""

from corpora.config import CVLConfig, IAMConfig, IAMHistDBConfig
from corpora.cvl import load_cvl, standardize_cvl_row
from corpora.historical import load_iam_histdb, standardize_historical_row
from corpora.iam import load_iam, standardize_iam_row
from corpora.schema import (
    DatasetSplit,
    HistoricalMeta,
    RecognitionMeta,
    RowMeta,
    WriterIdentityMeta,
)

__all__ = [
    "CVLConfig",
    "DatasetSplit",
    "HistoricalMeta",
    "IAMConfig",
    "IAMHistDBConfig",
    "RecognitionMeta",
    "RowMeta",
    "WriterIdentityMeta",
    "load_cvl",
    "load_iam",
    "load_iam_histdb",
    "standardize_cvl_row",
    "standardize_historical_row",
    "standardize_iam_row",
]
