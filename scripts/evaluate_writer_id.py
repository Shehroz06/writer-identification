"""Evaluates the writer-identification embedding model on held-out CVL +
Firemaker/CERUG test data, via the existing Phase 8
evaluation.identification (top-k accuracy, mAP), unmodified. Reports both
the trained (Circle-Loss fine-tuned) model and a frozen/untrained-DINOv2
baseline on the identical protocol -- the internal ablation from the
validation plan.

Query/gallery protocol: for each held-out (unseen) writer with at least 2
test samples, one sample is held out as the query and the rest become that
writer's gallery entries -- the standard leave-one-out retrieval protocol.

Not part of `handwriting_engine` -- a standalone evaluation-glue
script, per the "freeze the architecture" validation pass.

Hydra-driven, same config surface as `scripts.train corpus=writer_id`: the embedding
architecture (`embeddings.*`) must match whatever the checkpoint being
loaded was trained with, and `training.device` selects CPU/CUDA, e.g.:

    uv run python -m scripts.evaluate_writer_id training.device=cuda

Usage: uv run python -m scripts.evaluate_writer_id
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import hydra
import numpy as np
from datasets import load_dataset
from handwriting_engine.embeddings.config import EmbeddingConfig
from handwriting_engine.embeddings.model import EmbeddingModel
from handwriting_engine.evaluation.config import IdentificationConfig
from handwriting_engine.evaluation.identification import compute_identification_metrics
from handwriting_engine.training.checkpoint import load_checkpoint
from handwriting_engine.utils.device import resolve_device
from omegaconf import DictConfig, OmegaConf
from PIL.Image import Image as PILImage

from scripts._common import embed_all, encode_labels

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CVL_DIR = REPO_ROOT / "data" / "raw" / "cvl"
FIREMAKER_CERUG_DIR = REPO_ROOT / "data" / "raw" / "firemaker_cerug"
CHECKPOINT_PATH = REPO_ROOT / "models" / "checkpoints" / "writer_id" / "model.pt"
SPLIT_SEED = 42


def _load_test_rows() -> tuple[list[PILImage], list[str]]:
    cvl = load_dataset("imagefolder", data_dir=str(CVL_DIR))
    firemaker_cerug = load_dataset("imagefolder", data_dir=str(FIREMAKER_CERUG_DIR))

    images: list[PILImage] = []
    writer_ids: list[str] = []
    for row in cvl["test"]:
        images.append(row["image"])
        writer_ids.append(row["writer_id"])
    for row in firemaker_cerug["test"]:
        images.append(row["image"])
        writer_ids.append(row["writer_id"])
    return images, writer_ids


def _split_query_gallery(writer_ids: list[str]) -> tuple[list[int], list[int]]:
    """One query index per writer with >= 2 samples; every other sample from
    that writer becomes a gallery entry. Writers with only 1 test sample are
    skipped entirely (no query without a possible gallery match)."""
    by_writer: dict[str, list[int]] = {}
    for index, writer_id in enumerate(writer_ids):
        by_writer.setdefault(writer_id, []).append(index)

    rng = random.Random(SPLIT_SEED)
    query_indices: list[int] = []
    gallery_indices: list[int] = []
    for indices in by_writer.values():
        if len(indices) < 2:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        query_indices.append(shuffled[0])
        gallery_indices.extend(shuffled[1:])
    return query_indices, gallery_indices


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.training.device)
    embedding_config = EmbeddingConfig.model_validate(
        OmegaConf.to_container(cfg.embeddings, resolve=True)
    )

    images, writer_ids = _load_test_rows()
    labels, mapping = encode_labels(writer_ids)
    query_indices, gallery_indices = _split_query_gallery(writer_ids)
    logger.info(f"device: {device}")
    logger.info(
        f"Held-out test corpus: {len(images)} images, {len(mapping)} writers, "
        f"{len(query_indices)} queries, {len(gallery_indices)} gallery entries"
    )

    query_labels = np.array([labels[i] for i in query_indices], dtype=np.int_)
    gallery_labels = np.array([labels[i] for i in gallery_indices], dtype=np.int_)
    identification_config = IdentificationConfig(top_k_values=(1, 5, 10))

    def _evaluate(model: EmbeddingModel, label: str) -> None:
        all_embeddings = embed_all(model, images, embedding_config.backbone).numpy()
        query_embeddings = all_embeddings[query_indices]
        gallery_embeddings = all_embeddings[gallery_indices]
        metrics = compute_identification_metrics(
            query_embeddings,
            query_labels,
            gallery_embeddings,
            gallery_labels,
            identification_config,
        )
        logger.info(f"=== {label} ===")
        logger.info(f"top-k accuracy: {metrics.top_k_accuracy}")
        logger.info(f"mAP: {metrics.mean_average_precision:.4f}")

    if not cfg.skip_baseline:
        baseline_model = EmbeddingModel(embedding_config).to(device)
        baseline_model.eval()
        _evaluate(baseline_model, "Baseline: frozen/untrained DINOv2 embeddings")

    checkpoint_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise SystemExit(f"No trained checkpoint at {checkpoint_path}; run training first.")
    trained_model = EmbeddingModel(embedding_config).to(device)
    load_checkpoint(trained_model, checkpoint_path, map_location=str(device))
    trained_model.eval()
    _evaluate(trained_model, f"Trained: Circle-Loss fine-tuned embeddings ({checkpoint_path.name})")


if __name__ == "__main__":
    main()
