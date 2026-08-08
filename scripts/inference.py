"""Runs the trained embedding model on a single image: loads a checkpoint,
preprocesses the image the same way training did, and prints its embedding
vector.

This project's models are metric-learning *embedding* models (Circle Loss),
not classifiers -- there is no softmax "writer id" output baked into the
checkpoint. "Writer id prediction" here means a 1-nearest-neighbor lookup
against a gallery of known embeddings (the same principle
`evaluation.identification` uses for top-k/mAP), given via `--gallery`; a
gallery is a `.npz` file with `embeddings` (num_samples x embedding_dim,
L2-normalized) and `labels` (num_samples, string writer IDs) arrays -- build
one with a few lines of `scripts._common.embed_all` over a labeled split,
e.g.:

    from scripts._common import embed_all
    embeddings = embed_all(model, images, embedding_config.backbone).numpy()
    np.savez("gallery.npz", embeddings=embeddings, labels=np.array(writer_ids))

Without `--gallery`, only the embedding itself is produced.

Not part of `handwriting_engine` -- standalone inference glue, same
"freeze the architecture" spirit as scripts/train_*.py and
scripts/evaluate_*.py. Plain argparse (not Hydra) since this is a one-image,
one-shot CLI rather than a configurable experiment.

Usage:
    uv run python -m scripts.inference --image path/to/image.png
    uv run python -m scripts.inference --image path/to/image.png \\
        --checkpoint models/checkpoints/writer_id/best_model.pt \\
        --gallery models/checkpoints/writer_id/gallery.npz

Against an exported deployment/ bundle (see scripts/export_model.py), pass
its plain config.yaml instead of composing configs/:

    uv run python -m scripts.inference --image path/to/image.png \\
        --checkpoint deployment/best_model.pt --embedding-config deployment/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from handwriting_engine.embeddings.config import EmbeddingConfig
from handwriting_engine.embeddings.model import EmbeddingModel
from handwriting_engine.models.backbone import prepare_pixel_values
from handwriting_engine.training.checkpoint import load_checkpoint
from handwriting_engine.utils.device import resolve_device
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "checkpoints" / "writer_id" / "best_model.pt"


def load_embedding_config(explicit_path: Path | None) -> EmbeddingConfig:
    """Load the embedding architecture config either from a plain exported
    YAML (`--embedding-config`, e.g. a `deployment/config.yaml` written by
    `scripts/export_model.py`) or, by default, by composing this repo's own
    `configs/` the same way every other script here does."""
    if explicit_path is not None:
        with explicit_path.open() as fh:
            return EmbeddingConfig.model_validate(yaml.safe_load(fh))
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_DIR)):
        cfg = compose(config_name="config")
    return EmbeddingConfig.model_validate(OmegaConf.to_container(cfg.embeddings, resolve=True))


def embed_image(
    image_path: Path, checkpoint_path: Path, embedding_config: EmbeddingConfig, device_name: str
) -> tuple[NDArray[np.float64], torch.device]:
    if not checkpoint_path.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint_path}; train a model first "
            "(see README's 'Training and evaluating on real data')."
        )
    device = resolve_device(device_name)
    model = EmbeddingModel(embedding_config).to(device)
    load_checkpoint(model, checkpoint_path, map_location=str(device))
    model.eval()

    image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
    pixel_values = prepare_pixel_values([image], embedding_config.backbone).to(device)
    with torch.no_grad():
        embedding = model(pixel_values)[0].cpu().numpy().astype(np.float64)
    return embedding, device


def predict_writer(embedding: NDArray[np.float64], gallery_path: Path) -> tuple[str, float]:
    """1-nearest-neighbor lookup: predicts the gallery identity whose
    embedding has the highest cosine similarity to `embedding` (embeddings
    are L2-normalized, so this is a plain dot product -- same convention as
    `evaluation.identification.compute_identification_metrics`)."""
    gallery = np.load(gallery_path, allow_pickle=True)
    gallery_embeddings: NDArray[np.float64] = gallery["embeddings"]
    gallery_labels = gallery["labels"]
    similarities = gallery_embeddings @ embedding
    best_index = int(np.argmax(similarities))
    return str(gallery_labels[best_index]), float(similarities[best_index])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to a model checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--embedding-config",
        type=Path,
        default=None,
        help="Plain embeddings config YAML (e.g. deployment/config.yaml). Defaults to "
        "composing this repo's configs/ directory.",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        default=None,
        help="Optional .npz gallery (embeddings + labels arrays) for nearest-neighbor "
        "writer-id prediction. Without it, only the embedding is printed.",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    args = parser.parse_args()

    embedding_config = load_embedding_config(args.embedding_config)
    embedding, device = embed_image(args.image, args.checkpoint, embedding_config, args.device)

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"embedding size: {embedding.shape[0]}")
    print(f"embedding vector: {embedding.tolist()}")

    if args.gallery is None:
        print(
            "writer id prediction: skipped -- this model has no classifier, only embeddings. "
            "Pass --gallery <path.npz> for nearest-neighbor writer-id prediction (see this "
            "script's module docstring for how to build one)."
        )
        return

    if not args.gallery.exists():
        raise SystemExit(f"No gallery at {args.gallery}.")
    writer_id, similarity = predict_writer(embedding, args.gallery)
    print(f"writer id prediction: {writer_id} (cosine similarity={similarity:.4f})")


if __name__ == "__main__":
    main()
