"""Quantifies how much corpus/domain identity (CVL vs Firemaker/CERUG) is
separable in the embedding space, and how much it competes with genuine
writer identity during retrieval -- the follow-up to the engine's Phase 9
error analysis, which found that 0 of 50 top-1 identification errors ever
crossed a corpus boundary (anecdotal: 10 example pairs). This script makes
that finding quantitative, on the same held-out test set / query-gallery
protocol as `scripts/evaluate_writer_id.py`.

Two diagnostics:

(a) Linear probe: how well can corpus be predicted from an embedding alone?
    Run on both frozen/untrained-DINOv2 embeddings and a trained checkpoint's
    embeddings, to separate "already present in generic DINOv2 features" from
    "amplified by Circle-Loss writer-ID training." Reported as balanced
    accuracy + ROC-AUC (not raw accuracy -- the corpora are ~31:1 imbalanced
    in the test set), cross-validated with GroupKFold grouped by writer_id (a
    writer's own samples are too self-similar for sample-level CV folds to be
    valid -- would leak).

(b) Cross-corpus similarity diagnostic: for each query, compares its
    similarity to (i) its true match, (ii) the hardest same-corpus non-match
    foil, (iii) the hardest cross-corpus foil (structurally never a true
    match, since writer IDs never span corpora). If cross-corpus foils score
    reliably lower than same-corpus foils (Mann-Whitney U test), that's
    quantitative evidence corpus identity is a strong, separate axis the
    embedding space resolves before writer identity.

Usage: uv run python -m scripts.probe_corpus_separability [checkpoint_path=...] [skip_baseline=true]
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from handwriting_engine.embeddings.config import EmbeddingConfig
from handwriting_engine.embeddings.model import EmbeddingModel
from handwriting_engine.training.checkpoint import load_checkpoint
from handwriting_engine.utils.device import resolve_device
from omegaconf import DictConfig, OmegaConf
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from scripts._common import embed_all
from scripts.evaluate_writer_id import CHECKPOINT_PATH, _load_test_rows, _split_query_gallery

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _corpus_of(writer_id: str) -> str:
    return "cvl" if writer_id.startswith("cvl_") else "firemaker_cerug"


def _probe_corpus_separability(embeddings: np.ndarray, writer_ids: list[str], label: str) -> None:
    y = np.array([0 if _corpus_of(wid) == "cvl" else 1 for wid in writer_ids])
    groups = np.array(writer_ids)

    n_splits = min(5, len(set(writer_ids)))
    cv = GroupKFold(n_splits=n_splits)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    proba = cross_val_predict(
        clf, embeddings, y, groups=groups, cv=cv, method="predict_proba"
    )[:, 1]
    pred = (proba >= 0.5).astype(int)

    balanced_acc = balanced_accuracy_score(y, pred)
    auc = roc_auc_score(y, proba)
    logger.info(f"=== Corpus linear probe: {label} ===")
    logger.info(f"balanced accuracy: {balanced_acc:.4f}")
    logger.info(f"ROC-AUC: {auc:.4f}")


def _cross_corpus_similarity_diagnostic(
    embeddings: np.ndarray, writer_ids: list[str], query_indices: list[int], gallery_indices: list[int]
) -> None:
    gallery_embeddings = embeddings[gallery_indices]
    gallery_writer_ids = np.array([writer_ids[i] for i in gallery_indices])
    gallery_corpus = np.array([_corpus_of(wid) for wid in gallery_writer_ids])

    genuine_sims: list[float] = []
    same_corpus_foil_sims: list[float] = []
    cross_corpus_foil_sims: list[float] = []

    for query_index in query_indices:
        query_writer_id = writer_ids[query_index]
        query_corpus = _corpus_of(query_writer_id)
        query_embedding = embeddings[query_index]
        sims = gallery_embeddings @ query_embedding

        genuine_mask = gallery_writer_ids == query_writer_id
        same_corpus_mask = (gallery_corpus == query_corpus) & ~genuine_mask
        cross_corpus_mask = gallery_corpus != query_corpus

        if genuine_mask.any():
            genuine_sims.append(float(sims[genuine_mask].max()))
        if same_corpus_mask.any():
            same_corpus_foil_sims.append(float(sims[same_corpus_mask].max()))
        if cross_corpus_mask.any():
            cross_corpus_foil_sims.append(float(sims[cross_corpus_mask].max()))

    genuine = np.array(genuine_sims)
    same_corpus = np.array(same_corpus_foil_sims)
    cross_corpus = np.array(cross_corpus_foil_sims)

    statistic, p_value = mannwhitneyu(same_corpus, cross_corpus, alternative="greater")

    logger.info("=== Cross-corpus similarity diagnostic ===")
    logger.info(f"genuine match similarity:        mean={genuine.mean():.4f} median={np.median(genuine):.4f} (n={len(genuine)})")
    logger.info(f"same-corpus foil similarity:      mean={same_corpus.mean():.4f} median={np.median(same_corpus):.4f} (n={len(same_corpus)})")
    logger.info(f"cross-corpus foil similarity:     mean={cross_corpus.mean():.4f} median={np.median(cross_corpus):.4f} (n={len(cross_corpus)})")
    logger.info(f"Mann-Whitney U (same > cross): statistic={statistic:.1f} p={p_value:.2e}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.training.device)
    embedding_config = EmbeddingConfig.model_validate(
        OmegaConf.to_container(cfg.embeddings, resolve=True)
    )

    images, writer_ids = _load_test_rows()
    query_indices, gallery_indices = _split_query_gallery(writer_ids)
    logger.info(
        f"Held-out test corpus: {len(images)} images, {len(set(writer_ids))} writers, "
        f"{len(query_indices)} queries, {len(gallery_indices)} gallery entries"
    )

    if not cfg.skip_baseline:
        baseline_model = EmbeddingModel(embedding_config).to(device)
        baseline_model.eval()
        baseline_embeddings = embed_all(baseline_model, images, embedding_config.backbone).numpy()
        _probe_corpus_separability(baseline_embeddings, writer_ids, "baseline (frozen/untrained DINOv2)")

    checkpoint_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise SystemExit(f"No trained checkpoint at {checkpoint_path}; run training first.")
    trained_model = EmbeddingModel(embedding_config).to(device)
    load_checkpoint(trained_model, checkpoint_path, map_location=str(device))
    trained_model.eval()
    trained_embeddings = embed_all(trained_model, images, embedding_config.backbone).numpy()
    _probe_corpus_separability(trained_embeddings, writer_ids, f"trained ({checkpoint_path.name})")
    _cross_corpus_similarity_diagnostic(trained_embeddings, writer_ids, query_indices, gallery_indices)


if __name__ == "__main__":
    main()
