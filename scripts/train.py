"""Trains the embedding model on this project's registered corpus (writer_id)
-- corpus selection is a Hydra override, kept even with one entry so a second
corpus can be added later without a new script:

    uv run python -m scripts.train corpus=writer_id

This project consumes `handwriting_engine` (the core engine) as a dependency;
nothing here is engine code -- it's this downstream project's own
training-glue script.

Every other setting (epochs, batch composition, learning rate, device, dataloader
workers, checkpoint/resume behavior, ...) comes from configs/training/default.yaml
and can be overridden the same way, e.g.:

    uv run python -m scripts.train corpus=writer_id training.num_epochs=30 \\
        training.device=cuda training.dataloader.num_workers=4 \\
        training.dataloader.pin_memory=true

Adding a new corpus means writing one loader function in
scripts/corpora_registry.py and one configs/corpus/<name>.yaml file -- not a new
training script.

CPU-development sizing: the shipped defaults (5 epochs x 30 batches) were
empirically timed at ~7s/batch (batch = 8 identities x 4 samples) on the
reference CPU-only development machine. Override
`training.num_epochs`/`training.sampler.batches_per_epoch` upward once run on
a GPU.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from scripts._common import resolve_training_config, run_training
from scripts.corpora_registry import CORPUS_LOADERS

REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    corpus_name = cfg.corpus.name
    if corpus_name not in CORPUS_LOADERS:
        raise ValueError(
            f"Unknown corpus {corpus_name!r} -- registered corpora: "
            f"{sorted(CORPUS_LOADERS)} (add one in scripts/corpora_registry.py "
            "and configs/corpus/<name>.yaml)"
        )
    default_checkpoint_dir = REPO_ROOT / cfg.corpus.default_checkpoint_dir
    training_config = resolve_training_config(cfg, default_checkpoint_dir)
    images, writer_ids = CORPUS_LOADERS[corpus_name]()
    run_training(
        images,
        writer_ids,
        cfg,
        training_config,
        corpus_name=cfg.corpus.display_name,
    )


if __name__ == "__main__":
    main()
