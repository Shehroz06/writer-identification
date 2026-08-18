"""Shared fixtures for `api` tests: a FastAPI `TestClient` wired to an
injected `AppState` built around a tiny, randomly-initialized DINOv2 model
(not the real trained checkpoint) plus a small synthetic gallery, so no test
here needs the real checkpoint/gallery on disk."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from handwriting_engine.embeddings.config import EmbeddingConfig, EmbeddingHeadConfig
from handwriting_engine.embeddings.model import EmbeddingModel
from handwriting_engine.models.backbone import DINOv2Backbone
from handwriting_engine.models.config import BackboneConfig
from numpy.typing import NDArray
from transformers import Dinov2Config, Dinov2Model

from writer_identification.backend.adapter import IdentificationAdapter
from writer_identification.backend.api.app import create_app
from writer_identification.backend.api.state import AppState, build_app_state
from writer_identification.backend.config import IdentificationAppConfig

_IMAGE_SIZE = 28
_OUTPUT_DIM = 8


@pytest.fixture
def tiny_dinov2_model() -> Dinov2Model:
    config = Dinov2Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=_IMAGE_SIZE,
        patch_size=14,
        mlp_ratio=2,
    )
    return Dinov2Model(config)


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
    config = IdentificationAppConfig(embedding=embedding_config)
    return IdentificationAdapter(
        config,
        embedding_model=embedding_model,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
    )


@pytest.fixture
def app_state(tiny_dinov2_model: Dinov2Model, tmp_path: Path) -> AppState:
    rng = np.random.default_rng(0)
    gallery_embeddings = rng.normal(size=(5, _OUTPUT_DIM))
    gallery_labels = np.array([f"writer_{i}" for i in range(5)])
    adapter = _build_adapter(
        tiny_dinov2_model, gallery_embeddings=gallery_embeddings, gallery_labels=gallery_labels
    )
    return build_app_state(adapter=adapter, gallery_path=tmp_path / "gallery.npz")


@pytest.fixture
def app_state_without_gallery(tiny_dinov2_model: Dinov2Model, tmp_path: Path) -> AppState:
    adapter = _build_adapter(tiny_dinov2_model)
    return build_app_state(adapter=adapter, gallery_path=tmp_path / "gallery.npz")


@pytest.fixture
def client(app_state: AppState) -> Iterator[TestClient]:
    app = create_app(state=app_state)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_gallery(app_state_without_gallery: AppState) -> Iterator[TestClient]:
    app = create_app(state=app_state_without_gallery)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes() -> bytes:
    image: NDArray[np.uint8] = np.zeros((60, 60), dtype=np.uint8)
    image[10:50, 10:50] = 255
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()
