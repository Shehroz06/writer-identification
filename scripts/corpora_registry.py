"""Registry of loader functions for `scripts/train.py`'s `corpus=<name>` selection.

This child project (writer identification) only registers its own corpus.
Adding another one means writing a loader function here and a matching
`configs/corpus/<name>.yaml` file -- not a new training script.

Not part of `handwriting_engine` -- a standalone training-glue module in this
child project, consuming the engine as a dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from datasets import load_dataset
from PIL.Image import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parents[1]
CVL_DIR = REPO_ROOT / "data" / "raw" / "cvl"
FIREMAKER_CERUG_DIR = REPO_ROOT / "data" / "raw" / "firemaker_cerug"


def _load_writer_id_rows() -> tuple[list[PILImage], list[str]]:
    """Writer identification: CVL + Firemaker/CERUG combined."""
    cvl = load_dataset("imagefolder", data_dir=str(CVL_DIR))
    firemaker_cerug = load_dataset("imagefolder", data_dir=str(FIREMAKER_CERUG_DIR))

    images: list[PILImage] = []
    writer_ids: list[str] = []
    for row in cvl["train"]:
        images.append(row["image"])
        writer_ids.append(row["writer_id"])  # already namespaced "cvl_*"
    for row in firemaker_cerug["train"]:
        images.append(row["image"])
        writer_ids.append(row["writer_id"])  # already namespaced "cerug_*"/"firemaker_*"
    return images, writer_ids


CORPUS_LOADERS: dict[str, Callable[[], tuple[list[PILImage], list[str]]]] = {
    "writer_id": _load_writer_id_rows,
}
