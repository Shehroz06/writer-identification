"""Typed, validated configuration models for every dataset loader."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class HubDatasetConfig(BaseModel):
    """Configuration for a loader backed by a HuggingFace Hub dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hub_id: str
    cache_dir: Path
    """Local directory `datasets.load_dataset` caches the downloaded Hub
    dataset under; also the DVC-tracked location of the raw data on disk."""


class LocalDatasetConfig(BaseModel):
    """Configuration for a loader backed by a local, DVC-tracked directory that
    follows HuggingFace's `imagefolder` convention: one subdirectory per split
    (e.g. `train/`, `test/`), each containing images plus a `metadata.jsonl`
    with this dataset's extra columns (see the loader module's docstring for
    the exact columns each dataset expects)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root_dir: Path


class IAMConfig(HubDatasetConfig):
    """Configuration for the IAM Handwriting Database loader."""

    hub_id: str = "Teklia/IAM-line"
    cache_dir: Path = Path("data/raw/iam")


class CVLConfig(LocalDatasetConfig):
    """Configuration for the CVL Database (writer identification/verification) loader."""

    root_dir: Path = Path("data/raw/cvl")


class SigComp2011Config(HubDatasetConfig):
    """Configuration for the ICDAR 2011 SigComp signature verification loader."""

    hub_id: str = "1aurent/ICDAR-2011"
    cache_dir: Path = Path("data/raw/sigcomp2011")


class IAMHistDBConfig(LocalDatasetConfig):
    """Configuration for the IAM-HistDB historical handwriting loader."""

    root_dir: Path = Path("data/raw/iam_histdb")


class DatasetsConfig(BaseModel):
    """Aggregate configuration for every dataset loader."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    iam: IAMConfig = IAMConfig()
    cvl: CVLConfig = CVLConfig()
    sigcomp2011: SigComp2011Config = SigComp2011Config()
    iam_histdb: IAMHistDBConfig = IAMHistDBConfig()
