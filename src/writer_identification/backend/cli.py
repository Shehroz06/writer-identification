"""Command-line demo for the writer-identification adapter, using this child
project's own trained checkpoint and gallery by default.

Usage::

    uv run python -m writer_identification.backend.cli image.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from writer_identification.backend.adapter import IdentificationAdapter
from writer_identification.backend.config import IdentificationAppConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "checkpoints" / "writer_id" / "best_model.pt"
DEFAULT_GALLERY = REPO_ROOT / "models" / "checkpoints" / "writer_id" / "gallery.npz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank a handwriting image against a gallery of known writers."
    )
    parser.add_argument("image", help="Path to the query image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Trained checkpoint to load (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        default=DEFAULT_GALLERY,
        help=f"Gallery .npz to rank against (default: {DEFAULT_GALLERY}).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of ranked matches to print.")
    args = parser.parse_args(argv)

    raw_image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if raw_image is None:
        print("Error: could not read input image.", file=sys.stderr)
        return 1
    image = raw_image.astype(np.uint8)

    if not args.checkpoint.exists():
        print(
            f"Error: no checkpoint at {args.checkpoint} -- train one first "
            "(uv run python -m scripts.train corpus=writer_id) or pass "
            "--checkpoint explicitly.",
            file=sys.stderr,
        )
        return 1
    if not args.gallery.exists():
        print(
            f"Error: no gallery at {args.gallery} -- see "
            "src/writer_identification/backend/adapter.py's module "
            "docstring, or scripts/inference.py's, for how to build one.",
            file=sys.stderr,
        )
        return 1

    config = IdentificationAppConfig(checkpoint_path=args.checkpoint, gallery_path=args.gallery)
    adapter = IdentificationAdapter.from_config(config)
    matches = adapter.identify(image, top_k=args.top_k)

    for rank, match in enumerate(matches, start=1):
        print(f"{rank}. {match.label} (cosine similarity={match.similarity:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
