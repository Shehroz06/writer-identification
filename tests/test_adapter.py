"""Tests for writer_identification.backend.adapter.

Uses a tiny, randomly-initialized DINOv2 model injected via `EmbeddingModel`
(no network call) for ranking-logic and edge-case tests, with gallery
embeddings/labels also injected directly. The real-checkpoint loading path
(`IdentificationAppConfig.checkpoint_path`, no injected model) is covered
separately in test_checkpoint_wiring.py, since that needs the real trained
weights, not this file's tiny model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from handwriting_engine.embeddings.config import EmbeddingConfig, EmbeddingHeadConfig
from handwriting_engine.embeddings.model import EmbeddingModel
from handwriting_engine.models.backbone import DINOv2Backbone
from handwriting_engine.models.config import BackboneConfig
from handwriting_engine.preprocessing.config import PreprocessingConfig
from numpy.typing import NDArray
from transformers import Dinov2Model

from writer_identification.backend.adapter import IdentificationAdapter
from writer_identification.backend.config import IdentificationAppConfig

_IMAGE_SIZE = 28
_OUTPUT_DIM = 8


def _build_adapter(
    tiny_dinov2_model: Dinov2Model,
    gallery_embeddings: NDArray[np.float64] | None = None,
    gallery_labels: NDArray[np.str_] | None = None,
) -> IdentificationAdapter:
    backbone_config = BackboneConfig(image_size=_IMAGE_SIZE)
    backbone = DINOv2Backbone(backbone_config, model=tiny_dinov2_model)
    embedding_config = EmbeddingConfig(
        backbone=backbone_config, head=EmbeddingHeadConfig(output_dim=_OUTPUT_DIM)
    )
    embedding_model = EmbeddingModel(embedding_config, backbone=backbone)

    config = IdentificationAppConfig(
        preprocessing=PreprocessingConfig(), embedding=embedding_config
    )
    return IdentificationAdapter(
        config,
        embedding_model=embedding_model,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
    )


def _sample_image(fill: int, size: int = 60) -> NDArray[np.uint8]:
    image: NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)
    image[10:50, 10:50] = fill
    return image


def test_identify_raises_without_a_configured_gallery(tiny_dinov2_model: Dinov2Model) -> None:
    adapter = _build_adapter(tiny_dinov2_model)
    image = _sample_image(fill=200)

    with pytest.raises(ValueError, match="No gallery configured"):
        adapter.identify(image)


def test_identify_ranks_exact_match_first(tiny_dinov2_model: Dinov2Model) -> None:
    # Deliberately reuses one embedding_model (not _build_adapter twice --
    # that would construct two independently, randomly initialized heads,
    # making the two adapters' embeddings incomparable) so the query's own
    # embedding is genuinely present in its own gallery.
    backbone_config = BackboneConfig(image_size=_IMAGE_SIZE)
    backbone = DINOv2Backbone(backbone_config, model=tiny_dinov2_model)
    embedding_config = EmbeddingConfig(
        backbone=backbone_config, head=EmbeddingHeadConfig(output_dim=_OUTPUT_DIM)
    )
    embedding_model = EmbeddingModel(embedding_config, backbone=backbone)
    config = IdentificationAppConfig(
        preprocessing=PreprocessingConfig(), embedding=embedding_config
    )

    image = _sample_image(fill=200)
    query_embedding = (
        IdentificationAdapter(config, embedding_model=embedding_model)
        .embed(image)
        .numpy()
        .astype(np.float64)
    )

    # A gallery where one entry is literally the query's own embedding, plus
    # two decoys that can't score higher against it.
    gallery_embeddings = np.stack(
        [query_embedding * -1.0, query_embedding, np.roll(query_embedding, 1)]
    )
    gallery_labels = np.array(["decoy_a", "self", "decoy_b"])
    adapter = IdentificationAdapter(
        config,
        embedding_model=embedding_model,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
    )

    matches = adapter.identify(image, top_k=3)

    assert matches[0].label == "self"
    assert matches[0].similarity == pytest.approx(float(query_embedding @ query_embedding))


def test_identify_respects_top_k(tiny_dinov2_model: Dinov2Model) -> None:
    rng = np.random.default_rng(0)
    gallery_embeddings = rng.normal(size=(5, _OUTPUT_DIM))
    gallery_labels = np.array([f"id_{i}" for i in range(5)])
    adapter = _build_adapter(
        tiny_dinov2_model, gallery_embeddings=gallery_embeddings, gallery_labels=gallery_labels
    )
    image = _sample_image(fill=200)

    matches = adapter.identify(image, top_k=2)

    assert len(matches) == 2


def test_identify_handles_empty_gallery(tiny_dinov2_model: Dinov2Model) -> None:
    gallery_embeddings: NDArray[np.float64] = np.empty((0, _OUTPUT_DIM), dtype=np.float64)
    gallery_labels: NDArray[np.str_] = np.empty((0,), dtype=str)
    adapter = _build_adapter(
        tiny_dinov2_model, gallery_embeddings=gallery_embeddings, gallery_labels=gallery_labels
    )
    image = _sample_image(fill=200)

    matches = adapter.identify(image)

    assert matches == []


def test_identify_handles_single_candidate_gallery(tiny_dinov2_model: Dinov2Model) -> None:
    gallery_embeddings = np.zeros((1, _OUTPUT_DIM), dtype=np.float64)
    gallery_embeddings[0, 0] = 1.0
    gallery_labels = np.array(["only_one"])
    adapter = _build_adapter(
        tiny_dinov2_model, gallery_embeddings=gallery_embeddings, gallery_labels=gallery_labels
    )
    image = _sample_image(fill=200)

    matches = adapter.identify(image, top_k=5)

    assert len(matches) == 1
    assert matches[0].label == "only_one"


def test_identify_breaks_ties_by_original_gallery_order(tiny_dinov2_model: Dinov2Model) -> None:
    """Two gallery entries with identical embeddings tie in similarity
    against any query -- a stable sort must keep them in their original
    relative order rather than shuffling arbitrarily."""
    identical_vector = np.full(_OUTPUT_DIM, 0.5, dtype=np.float64)
    gallery_embeddings = np.stack([identical_vector, identical_vector])
    gallery_labels = np.array(["first", "second"])
    adapter = _build_adapter(
        tiny_dinov2_model, gallery_embeddings=gallery_embeddings, gallery_labels=gallery_labels
    )
    image = _sample_image(fill=200)

    matches = adapter.identify(image, top_k=2)

    assert [match.label for match in matches] == ["first", "second"]
    assert matches[0].similarity == pytest.approx(matches[1].similarity)


def test_injected_embedding_model_takes_priority_over_checkpoint_path(
    tiny_dinov2_model: Dinov2Model,
) -> None:
    """An explicitly injected `embedding_model` must skip checkpoint loading
    entirely -- proven by pointing `checkpoint_path` at a nonexistent file
    and confirming construction still succeeds."""
    backbone_config = BackboneConfig(image_size=_IMAGE_SIZE)
    backbone = DINOv2Backbone(backbone_config, model=tiny_dinov2_model)
    embedding_config = EmbeddingConfig(
        backbone=backbone_config, head=EmbeddingHeadConfig(output_dim=_OUTPUT_DIM)
    )
    embedding_model = EmbeddingModel(embedding_config, backbone=backbone)
    config = IdentificationAppConfig(
        embedding=embedding_config,
        checkpoint_path=Path("/nonexistent/path/does-not-exist.pt"),
    )

    adapter = IdentificationAdapter(config, embedding_model=embedding_model)

    image = _sample_image(fill=200)
    embedding = adapter.embed(image)
    assert embedding.shape[0] == _OUTPUT_DIM
