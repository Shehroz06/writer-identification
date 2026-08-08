"""Shared fixtures for dataset-loader tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def make_tiny_image() -> Callable[[], Image.Image]:
    """Factory for a minimal valid grayscale image, standing in for a real
    scanned handwriting sample -- loader logic never inspects pixel content,
    only passes the image feature through."""

    def _make() -> Image.Image:
        return Image.fromarray(np.zeros((4, 4), dtype=np.uint8))

    return _make
