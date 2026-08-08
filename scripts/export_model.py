"""Exports the minimal set of files needed to run inference with a trained
checkpoint, separate from the rest of the repo/training infrastructure:

    deployment/
        best_model.pt   the checkpoint itself (renamed for a stable, predictable name)
        config.yaml      the embedding architecture config needed to reconstruct EmbeddingModel
        labels.json      the int-label -> writer_id mapping, if the training run wrote one
        inference.py     a copy of scripts/inference.py, for reference/portability

Still requires the `handwriting-engine` package installed wherever this
bundle is used -- this is not a zero-dependency artifact, just the minimal
*data* files (weights, config, labels) plus a copy of the runtime script,
so the bundle is self-describing without carrying the whole repo.

Not part of `handwriting_engine` -- standalone deployment glue, same
"freeze the architecture" spirit as the rest of scripts/. Plain argparse
(not Hydra), matching scripts/inference.py.

Usage:
    uv run python -m scripts.export_model --checkpoint models/checkpoints/writer_id/best_model.pt
    uv run python -m scripts.export_model \\
        --checkpoint models/checkpoints/signature_forgery/best_model.pt \\
        --output deployment/signature_forgery
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from handwriting_engine.embeddings.config import EmbeddingConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "deployment"


def _load_embedding_config() -> EmbeddingConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_DIR)):
        cfg = compose(config_name="config")
    return EmbeddingConfig.model_validate(OmegaConf.to_container(cfg.embeddings, resolve=True))


def export(checkpoint_path: Path, output_dir: Path) -> None:
    if not checkpoint_path.exists():
        raise SystemExit(f"No checkpoint at {checkpoint_path}; train a model first (see README).")
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_path, output_dir / "best_model.pt")

    embedding_config = _load_embedding_config()
    with (output_dir / "config.yaml").open("w") as fh:
        yaml.safe_dump(embedding_config.model_dump(mode="json"), fh, sort_keys=False)

    labels_path = checkpoint_path.parent / "labels.json"
    if labels_path.exists():
        shutil.copy2(labels_path, output_dir / "labels.json")
    else:
        print(
            f"note: no {labels_path} found -- skipping labels.json "
            "(train with training.checkpoint_dir set to produce one)"
        )

    shutil.copy2(REPO_ROOT / "scripts" / "inference.py", output_dir / "inference.py")

    print(f"\nExported to {output_dir}:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")
    print(
        f"\nRun inference against this bundle with:\n"
        f"  uv run python -m scripts.inference --image <path> "
        f"--checkpoint {output_dir / 'best_model.pt'} "
        f"--embedding-config {output_dir / 'config.yaml'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained checkpoint to export "
        "(e.g. models/checkpoints/writer_id/best_model.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the deployment bundle (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
