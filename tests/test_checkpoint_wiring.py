"""The first genuine end-to-end test of the production code path: real
checkpoint, real gallery, `preprocessing=None` (unlike test_adapter.py,
which only ever uses a tiny injected model with `preprocessing=
PreprocessingConfig()` explicitly set). Skipped when the checkpoint/gallery
aren't present on disk (e.g. CI), since both are too large to commit."""

from __future__ import annotations

import cv2
import pytest

from writer_identification.backend.adapter import IdentificationAdapter
from writer_identification.backend.cli import DEFAULT_CHECKPOINT, DEFAULT_GALLERY
from writer_identification.backend.config import IdentificationAppConfig

pytestmark = pytest.mark.skipif(
    not DEFAULT_CHECKPOINT.exists() or not DEFAULT_GALLERY.exists(),
    reason="requires the trained checkpoint + gallery on disk",
)


def test_identify_against_real_gallery_finds_writer_via_self_match() -> None:
    config = IdentificationAppConfig(
        checkpoint_path=DEFAULT_CHECKPOINT, gallery_path=DEFAULT_GALLERY
    )
    adapter = IdentificationAdapter.from_config(config)
    image = cv2.imread("data/raw/cvl/train/cvl_0001_1.png", cv2.IMREAD_GRAYSCALE)

    matches = adapter.identify(image, top_k=1)

    assert matches[0].label == "cvl_0001"
    assert matches[0].similarity == pytest.approx(1.0, abs=1e-4)
