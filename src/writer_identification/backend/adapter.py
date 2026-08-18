"""Writer identification: a child project built on the shared engine --
composes preprocessing and the embedding model with a nearest-neighbor
gallery lookup into a single `.identify(image)` call, with no changes to the
engine itself.

Generalizes the ad-hoc gallery lookup `scripts/inference.py --gallery`
performs into the adapter pattern -- any project needing "rank one image
against a set of known identities" can reuse this, not just writer ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch
from handwriting_engine.adapters.base import EngineAdapter
from handwriting_engine.adapters.embedding import embed_image
from handwriting_engine.embeddings.model import EmbeddingModel
from handwriting_engine.models.backbone import prepare_pixel_values
from handwriting_engine.training.checkpoint import load_checkpoint
from numpy.typing import NDArray

from writer_identification.backend.config import IdentificationAppConfig


@dataclass(frozen=True)
class IdentificationMatch:
    """One ranked gallery candidate for a query image."""

    label: str
    similarity: float


class IdentificationAdapter(EngineAdapter[IdentificationAppConfig]):
    """Embeds one image and ranks it against a gallery of known identities by
    cosine similarity -- the metric-learning equivalent of a classifier's
    top-k prediction, since this project's models have no softmax output."""

    def __init__(
        self,
        config: IdentificationAppConfig,
        embedding_model: EmbeddingModel | None = None,
        gallery_embeddings: NDArray[np.float64] | None = None,
        gallery_labels: NDArray[np.str_] | None = None,
    ) -> None:
        self._config = config
        if embedding_model is not None:
            self._embedding_model = embedding_model
        else:
            self._embedding_model = EmbeddingModel(config.embedding)
            if config.checkpoint_path is not None:
                load_checkpoint(self._embedding_model, config.checkpoint_path)
        self._embedding_model.eval()

        if gallery_embeddings is not None and gallery_labels is not None:
            self._gallery_embeddings: NDArray[np.float64] | None = gallery_embeddings
            self._gallery_labels: NDArray[np.str_] | None = gallery_labels
        elif config.gallery_path is not None:
            gallery = np.load(config.gallery_path, allow_pickle=True)
            self._gallery_embeddings = gallery["embeddings"]
            self._gallery_labels = gallery["labels"]
        else:
            self._gallery_embeddings = None
            self._gallery_labels = None

    @classmethod
    def from_config(cls, config: IdentificationAppConfig) -> Self:
        return cls(config)

    def embed(self, image: NDArray[np.uint8]) -> torch.Tensor:
        """Embed a single image -- usable standalone (e.g. to build a
        gallery) even without a gallery configured.

        With `config.preprocessing` at its default of `None`, skips the
        engine's preprocessing pipeline entirely (raw grayscale straight to
        the backbone), matching the distribution training/evaluation
        actually use. Only runs preprocessing if a `PreprocessingConfig` is
        explicitly set -- see `IdentificationAppConfig.preprocessing`'s
        docstring for why that path is currently unvalidated."""
        if self._config.preprocessing is None:
            pixel_values = prepare_pixel_values([image], self._config.embedding.backbone)
            with torch.no_grad():
                embedding: torch.Tensor = self._embedding_model(pixel_values)[0]
            return embedding
        return embed_image(
            self._embedding_model,
            image,
            self._config.preprocessing,
            self._config.embedding.backbone,
        )

    def identify(self, image: NDArray[np.uint8], top_k: int = 5) -> list[IdentificationMatch]:
        """Ranks `image` against the configured gallery by cosine similarity,
        highest first (ties broken by original gallery order). Raises if no
        gallery was configured."""
        if self._gallery_embeddings is None or self._gallery_labels is None:
            raise ValueError(
                "No gallery configured -- set config.gallery_path or pass "
                "gallery_embeddings/gallery_labels to IdentificationAdapter."
            )
        embedding = self.embed(image).numpy().astype(np.float64)
        similarities = self._gallery_embeddings @ embedding
        ranked_indices = np.argsort(-similarities, kind="stable")[:top_k]
        return [
            IdentificationMatch(
                label=str(self._gallery_labels[index]),
                similarity=float(similarities[index]),
            )
            for index in ranked_indices
        ]

    @property
    def gallery_size(self) -> int:
        return 0 if self._gallery_embeddings is None else len(self._gallery_embeddings)

    def enroll(self, image: NDArray[np.uint8], writer_id: str) -> None:
        """Embed `image` (same code path as querying, so enrolled samples
        and future queries live in the same distribution) and append it to
        the in-memory gallery, initializing an empty gallery first if none
        was configured."""
        embedding = self.embed(image).numpy().astype(np.float64)
        if self._gallery_embeddings is None or self._gallery_labels is None:
            self._gallery_embeddings = embedding.reshape(1, -1)
            self._gallery_labels = np.array([writer_id])
        else:
            self._gallery_embeddings = np.vstack([self._gallery_embeddings, embedding])
            self._gallery_labels = np.append(self._gallery_labels, writer_id)

    def save_gallery(self, path: Path) -> None:
        """Write the current in-memory gallery to `path`, using the same
        `embeddings`/`labels` `.npz` convention `scripts/build_gallery.py`
        uses -- byte-for-byte compatible with the existing format."""
        embeddings = (
            self._gallery_embeddings
            if self._gallery_embeddings is not None
            else np.empty((0, 0), dtype=np.float64)
        )
        labels = (
            self._gallery_labels
            if self._gallery_labels is not None
            else np.empty((0,), dtype=str)
        )
        np.savez(path, embeddings=embeddings, labels=labels)
