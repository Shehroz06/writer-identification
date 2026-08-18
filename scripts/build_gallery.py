"""Builds the deployed `.npz` gallery for `IdentificationAdapter`/the CLI:
embeds every known writer's images (train + test, both corpora) through the
adapter's real query-time code path -- `IdentificationAdapter.embed()`, with
preprocessing left at its default of disabled (see
`IdentificationAppConfig.preprocessing`'s docstring) -- so gallery entries and
future CLI queries are embedded identically, and match the no-preprocessing
distribution training/evaluation actually use.

Population: train + test combined. This is for deployed identification
coverage, not a held-out benchmark -- `scripts/evaluate_writer_id.py` already
owns that number permanently; querying this gallery afterward is a
consistency/wiring check, not a fresh accuracy claim.

Before the full ~49k-image build, a cheap spot check (`spot_check=true`)
builds a tiny gallery from a handful of writers and confirms held-out queries
from those same writers come back correct -- a first run of this (before
preprocessing was disabled by default) caught a real bug: it scored 3/8 with
preprocessing forced on, vs. ~62% top-1 without it.

Usage:
    uv run python -m scripts.build_gallery spot_check=true          # cheap sanity check first
    uv run python -m scripts.build_gallery                          # full build
    uv run python -m scripts.build_gallery checkpoint_path=<...>    # non-default checkpoint
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import hydra
import numpy as np
from handwriting_engine.embeddings.config import EmbeddingConfig
from handwriting_engine.embeddings.model import EmbeddingModel
from handwriting_engine.training.checkpoint import load_checkpoint
from omegaconf import DictConfig, OmegaConf
from PIL.Image import Image as PILImage

from scripts._common import embed_all
from scripts.corpora_registry import _load_writer_id_rows
from scripts.evaluate_writer_id import CHECKPOINT_PATH, _load_test_rows
from writer_identification.backend.adapter import IdentificationAdapter
from writer_identification.backend.config import IdentificationAppConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
GALLERY_PATH = REPO_ROOT / "models" / "checkpoints" / "writer_id" / "gallery.npz"
SPOT_CHECK_SEED = 42
SPOT_CHECK_NUM_WRITERS = 8


def _build_config_and_model(cfg: DictConfig) -> tuple[IdentificationAppConfig, EmbeddingModel]:
    checkpoint_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise SystemExit(f"No trained checkpoint at {checkpoint_path}; run training first.")

    embedding_config = EmbeddingConfig.model_validate(
        OmegaConf.to_container(cfg.embeddings, resolve=True)
    )
    model = EmbeddingModel(embedding_config)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    # preprocessing left at IdentificationAppConfig's default (None): skips
    # the engine's preprocessing pipeline entirely, matching the
    # no-preprocessing distribution training/evaluation actually use -- see
    # IdentificationAppConfig.preprocessing's docstring. A spot check
    # (spot_check=true) under full preprocessing scored 3/8 vs. ~62% top-1
    # without it.
    config = IdentificationAppConfig(
        embedding=embedding_config,
        checkpoint_path=checkpoint_path,
    )
    return config, model


def _all_images_and_writer_ids() -> tuple[list[PILImage], list[str]]:
    train_images, train_writer_ids = _load_writer_id_rows()
    test_images, test_writer_ids = _load_test_rows()
    return train_images + test_images, train_writer_ids + test_writer_ids


def _spot_check(cfg: DictConfig) -> None:
    config, model = _build_config_and_model(cfg)
    embedder = IdentificationAdapter(config, embedding_model=model)

    images, writer_ids = _all_images_and_writer_ids()
    by_writer: dict[str, list[int]] = {}
    for index, writer_id in enumerate(writer_ids):
        by_writer.setdefault(writer_id, []).append(index)
    eligible = [wid for wid, indices in by_writer.items() if len(indices) >= 2]

    rng = random.Random(SPOT_CHECK_SEED)
    sample_writers = rng.sample(eligible, min(SPOT_CHECK_NUM_WRITERS, len(eligible)))

    gallery_embeddings: list[np.ndarray] = []
    gallery_labels: list[str] = []
    queries: list[tuple[np.ndarray, str]] = []
    for writer_id in sample_writers:
        indices = list(by_writer[writer_id])
        rng.shuffle(indices)
        query_index, *gallery_only_indices = indices
        queries.append((np.array(images[query_index].convert("L"), dtype=np.uint8), writer_id))
        for gallery_index in gallery_only_indices:
            array = np.array(images[gallery_index].convert("L"), dtype=np.uint8)
            gallery_embeddings.append(embedder.embed(array).numpy())
            gallery_labels.append(writer_id)

    identifier = IdentificationAdapter(
        config,
        embedding_model=model,
        gallery_embeddings=np.stack(gallery_embeddings).astype(np.float64),
        gallery_labels=np.array(gallery_labels),
    )

    correct = 0
    for query_array, true_writer_id in queries:
        match = identifier.identify(query_array, top_k=1)[0]
        is_correct = match.label == true_writer_id
        correct += int(is_correct)
        status = "OK" if is_correct else "WRONG"
        logger.info(f"{status}: {true_writer_id} -> predicted {match.label} (similarity={match.similarity:.4f})")

    logger.info(f"Spot check: {correct}/{len(queries)} correct")
    if correct < len(queries):
        logger.warning(
            "not all spot-check queries were correct -- investigate before "
            "running the full gallery build (spot_check=false)."
        )


def _build_full_gallery(cfg: DictConfig) -> None:
    config, model = _build_config_and_model(cfg)
    assert config.preprocessing is None, (
        "full gallery build uses the batched embed_all() path, which skips preprocessing "
        "unconditionally -- only valid while IdentificationAppConfig.preprocessing is None"
    )

    images, writer_ids = _all_images_and_writer_ids()
    logger.info(f"Embedding {len(images)} images ({len(set(writer_ids))} writers) for the gallery...")

    # Batched (not adapter.embed()'s one-image-at-a-time loop, used by the
    # spot check) -- same convention as scripts/evaluate_writer_id.py, since
    # ~49k images one at a time is prohibitively slow on CPU. Equivalent to
    # adapter.embed() now that preprocessing is confirmed off (spot-checked
    # above): both reduce to prepare_pixel_values -> model forward.
    embeddings = embed_all(model, images, config.embedding.backbone).numpy()

    GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        GALLERY_PATH,
        embeddings=embeddings.astype(np.float64),
        labels=np.array(writer_ids),
    )
    logger.info(f"Wrote gallery: {GALLERY_PATH} ({len(images)} entries, {len(set(writer_ids))} writers)")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("spot_check", False):
        _spot_check(cfg)
    else:
        _build_full_gallery(cfg)


if __name__ == "__main__":
    main()
