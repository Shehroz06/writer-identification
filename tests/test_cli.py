"""Tests for writer_identification.backend.cli's argument
handling and error paths -- doesn't exercise the real embedding-model path
(that needs the actual trained checkpoint; see test_checkpoint_wiring.py and
this project's manual CLI smoke test for that)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from writer_identification.backend.cli import main


def test_main_returns_error_for_missing_image_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.png"
    exit_code = main([str(missing)])

    assert exit_code == 1


def test_main_returns_error_for_missing_checkpoint(tmp_path: Path) -> None:
    image = tmp_path / "query.png"
    cv2.imwrite(str(image), np.zeros((32, 32), dtype=np.uint8))
    missing_checkpoint = tmp_path / "no-such-checkpoint.pt"

    exit_code = main([str(image), "--checkpoint", str(missing_checkpoint)])

    assert exit_code == 1


def test_main_returns_error_for_missing_gallery(tmp_path: Path) -> None:
    """A real checkpoint but no gallery must fail with a clear message
    rather than reaching the (much slower) model-loading path at all."""
    image = tmp_path / "query.png"
    cv2.imwrite(str(image), np.zeros((32, 32), dtype=np.uint8))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()  # existence is all the CLI checks before the gallery
    missing_gallery = tmp_path / "no-such-gallery.npz"

    exit_code = main(
        [
            str(image),
            "--checkpoint",
            str(checkpoint),
            "--gallery",
            str(missing_gallery),
        ]
    )

    assert exit_code == 1
