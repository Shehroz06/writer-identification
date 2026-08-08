"""Shared fixtures for `applications` tests: a tiny, randomly-initialized
DINOv2 model (not a downloaded pretrained checkpoint) so tests exercise the
verification adapter without a network call."""

from __future__ import annotations

import pytest
from transformers import Dinov2Config, Dinov2Model


@pytest.fixture
def tiny_dinov2_model() -> Dinov2Model:
    config = Dinov2Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
        mlp_ratio=2,
    )
    return Dinov2Model(config)
