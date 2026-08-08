"""Unit tests for corpora.schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from corpora.schema import (
    DatasetSplit,
    HistoricalMeta,
    RecognitionMeta,
    SignatureMeta,
    WriterIdentityMeta,
)


def test_recognition_meta_defaults_writer_id_to_none() -> None:
    meta = RecognitionMeta(transcription="hello", split=DatasetSplit.TRAIN)
    assert meta.writer_id is None


def test_writer_identity_meta_requires_writer_id() -> None:
    with pytest.raises(ValidationError):
        WriterIdentityMeta(split=DatasetSplit.TRAIN)  # type: ignore[call-arg]


def test_signature_meta_round_trips_is_genuine() -> None:
    meta = SignatureMeta(writer_id="1", is_genuine=False, split=DatasetSplit.TEST)
    assert meta.is_genuine is False


def test_historical_meta_allows_missing_transcription() -> None:
    meta = HistoricalMeta(collection="parzival", split=DatasetSplit.TRAIN)
    assert meta.transcription is None


def test_row_meta_models_are_frozen_and_reject_extra_fields() -> None:
    meta = RecognitionMeta(transcription="hello", split=DatasetSplit.TRAIN)
    with pytest.raises(ValidationError):
        meta.transcription = "changed"

    with pytest.raises(ValidationError):
        RecognitionMeta(transcription="hello", split=DatasetSplit.TRAIN, unexpected=1)  # type: ignore[call-arg]
