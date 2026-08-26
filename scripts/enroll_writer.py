"""Bulk/offline enrollment of a writer's samples straight from local image
files, without needing the API server running -- same `IdentificationAdapter
.enroll()`/`.save_gallery()` path the `/enroll` endpoint uses, so the result
is byte-for-byte the same gallery format.

Plain argparse (not Hydra), same one-shot-CLI spirit as scripts/inference.py.

Safety note: this mutates the gallery file in place -- the same file the
deployed service reads from. Back it up before your first enrollment
(`cp gallery.npz gallery.npz.bak`) if you want the original recoverable.

Usage:
    uv run python -m scripts.enroll_writer --writer-id alice \\
        path/to/sample1.png path/to/sample2.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from writer_identification.backend.adapter import IdentificationAdapter
from writer_identification.backend.cli import DEFAULT_CHECKPOINT, DEFAULT_GALLERY
from writer_identification.backend.config import IdentificationAppConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enroll one or more image samples for a writer into the gallery."
    )
    parser.add_argument("images", nargs="+", type=Path, help="Paths to the sample images.")
    parser.add_argument("--writer-id", required=True, help="Identity label for these samples.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Trained checkpoint to embed with (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        default=DEFAULT_GALLERY,
        help=f"Gallery .npz to enroll into and save back to (default: {DEFAULT_GALLERY}).",
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"Error: no checkpoint at {args.checkpoint}.", file=sys.stderr)
        return 1

    gallery_path = args.gallery if args.gallery.exists() else None
    config = IdentificationAppConfig(checkpoint_path=args.checkpoint, gallery_path=gallery_path)
    adapter = IdentificationAdapter.from_config(config)

    for image_path in args.images:
        raw_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if raw_image is None:
            print(f"Error: could not read {image_path}.", file=sys.stderr)
            return 1
        adapter.enroll(raw_image.astype(np.uint8), args.writer_id)

    adapter.save_gallery(args.gallery)
    print(
        f"Enrolled {len(args.images)} sample(s) for '{args.writer_id}' into {args.gallery} "
        f"(gallery now has {adapter.gallery_size} entries)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
