"""Shared glue for `scripts/train.py`'s generic training entrypoint.

Not engine code -- this child project consumes `handwriting_engine` as a
dependency; this module holds the corpus-agnostic dataset-wrapping/collate/
training-loop-driving logic local to this project, so a new corpus
(registered in `scripts/corpora_registry.py`) only needs its own loader
function, not a copy of this mechanics.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from handwriting_engine.embeddings.config import EmbeddingConfig
from handwriting_engine.embeddings.model import EmbeddingModel
from handwriting_engine.models.backbone import prepare_pixel_values
from handwriting_engine.models.config import BackboneConfig
from handwriting_engine.training.config import TrainingConfig
from handwriting_engine.training.loop import EpochResult, train
from handwriting_engine.training.sampler import PKSampler
from handwriting_engine.utils.device import resolve_device
from handwriting_engine.utils.seeding import set_global_seed
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from PIL.Image import Image as PILImage
from torch.utils.data import DataLoader, Dataset


class ImageLabelDataset(Dataset[tuple[NDArray[np.uint8], int]]):
    """Wraps parallel sequences of PIL images and integer identity labels
    (typically pooled from more than one combined HF dataset)."""

    def __init__(self, images: Sequence[PILImage], labels: Sequence[int]) -> None:
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[NDArray[np.uint8], int]:
        array = np.array(self._images[index].convert("L"), dtype=np.uint8)
        return array, self._labels[index]


def make_collate_fn(
    backbone_config: BackboneConfig,
) -> Callable[[list[tuple[NDArray[np.uint8], int]]], tuple[torch.Tensor, torch.Tensor]]:
    """Batch-level collate: resizes/normalizes raw grayscale arrays into the
    tensor `EmbeddingModel` expects (`models.backbone.prepare_pixel_values`),
    since that has to happen per-batch, not per-sample."""

    def collate(
        batch: list[tuple[NDArray[np.uint8], int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        pixel_values = prepare_pixel_values(images, backbone_config)
        return pixel_values, labels

    return collate


def encode_labels(writer_ids: Sequence[str]) -> tuple[list[int], dict[str, int]]:
    """Maps string writer IDs (already namespaced by source, e.g. `cvl_0001`)
    onto dense integer labels for `PKSampler`/the metric-learning losses."""
    unique = sorted(set(writer_ids))
    mapping = {writer_id: index for index, writer_id in enumerate(unique)}
    return [mapping[writer_id] for writer_id in writer_ids], mapping


def print_split_summary(name: str, writer_ids: Sequence[str]) -> None:
    print(f"{name}: {len(writer_ids)} samples, {len(set(writer_ids))} writers")


def already_prepared(output_dir: Path) -> bool:
    """Whether `scripts.prepare_*`'s output already exists at `output_dir`
    (both `train/metadata.jsonl` and `test/metadata.jsonl` present).

    Lets each prepare script skip straight back to a no-op when re-run
    against an already-prepared `data/raw/` -- the common case after a
    disconnect/restart on a session where `data/raw/` is Drive-persisted
    (see `scripts/colab_bootstrap.py`), where re-deriving already-prepared
    data would just waste CPU time and (for archives re-extracted from
    `data/incoming/`) I/O for no benefit.
    """
    return (output_dir / "train" / "metadata.jsonl").exists() and (
        output_dir / "test" / "metadata.jsonl"
    ).exists()


def write_labels(checkpoint_dir: Path, mapping: dict[str, int]) -> Path:
    """Persist the writer_id->int label mapping `encode_labels` computed for
    this training run to `checkpoint_dir/labels.json`, as `{int: writer_id}`
    (the direction useful for turning a predicted label back into a writer
    ID). Without this, the mapping only ever existed in-memory for the
    duration of one training run -- unrecoverable afterward, which makes a
    checkpoint alone insufficient for `scripts/export_model.py`/
    `scripts/inference.py` to report which writer a label corresponds to.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "labels.json"
    reverse_mapping = {str(index): writer_id for writer_id, index in mapping.items()}
    with path.open("w") as fh:
        json.dump(reverse_mapping, fh, indent=2)
    return path


def hf_row_images_and_writers(
    rows: Any, image_key: str = "image", writer_key: str = "writer_id"
) -> tuple[list[PILImage], list[str]]:
    """Pull parallel (image, writer_id) lists out of one HF `Dataset` split."""
    images = [row[image_key] for row in rows]
    writer_ids = [row[writer_key] for row in rows]
    return images, writer_ids


def embed_all(
    model: EmbeddingModel,
    images: Sequence[PILImage],
    backbone_config: BackboneConfig,
    batch_size: int = 32,
) -> torch.Tensor:
    """Batch-embeds a list of PIL images with `model`.

    Same device convention as `evaluation.benchmark.embed_dataset`: runs on
    whichever device `model`'s parameters already live on (move the model
    there first via `.to(device)`); always returns embeddings on CPU so both
    the torch-based and numpy-based evaluation scripts can consume them
    without an extra `.cpu()` at every call site.
    """
    device = next(model.parameters()).device
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            arrays: list[NDArray[np.uint8]] = [
                np.array(image.convert("L"), dtype=np.uint8) for image in batch
            ]
            pixel_values = prepare_pixel_values(arrays, backbone_config).to(device)
            embeddings.append(model(pixel_values).cpu())
    return torch.cat(embeddings, dim=0)


def resolve_training_config(cfg: DictConfig, default_checkpoint_dir: Path) -> TrainingConfig:
    """Validate the composed `training` config group into a `TrainingConfig`,
    same `OmegaConf.to_container` + `model_validate` pattern as `api/config.py`.

    Fills in `checkpoint_dir` with this script's own default when the user
    hasn't overridden it via the CLI (e.g. `training.checkpoint_dir=...`), so
    each training script keeps a sensible per-task default while staying
    fully overridable.
    """
    training_dict = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_dict, dict)
    if training_dict.get("checkpoint_dir") is None:
        training_dict["checkpoint_dir"] = str(default_checkpoint_dir)
    return TrainingConfig.model_validate(training_dict)


def run_training(
    images: Sequence[PILImage],
    writer_ids: Sequence[str],
    cfg: DictConfig,
    training_config: TrainingConfig,
    corpus_name: str,
) -> list[EpochResult]:
    """Shared entrypoint for the real-data training scripts: seeds every
    global RNG from `cfg.seed`, builds the P-K sampled DataLoader + embedding
    model from the composed Hydra config, and runs
    `handwriting_engine.training.loop.train`, printing progress as it goes.
    """
    set_global_seed(cfg.seed)
    print_split_summary(corpus_name, writer_ids)

    labels, mapping = encode_labels(writer_ids)
    if training_config.checkpoint_dir is not None:
        write_labels(training_config.checkpoint_dir, mapping)

    embedding_config = EmbeddingConfig.model_validate(
        OmegaConf.to_container(cfg.embeddings, resolve=True)
    )
    dataset = ImageLabelDataset(images, labels)
    sampler = PKSampler(
        labels,
        num_identities_per_batch=training_config.sampler.num_identities_per_batch,
        num_samples_per_identity=training_config.sampler.num_samples_per_identity,
        batches_per_epoch=training_config.sampler.batches_per_epoch,
        seed=cfg.seed,
    )
    # DataLoader's generic type below reflects the Dataset's per-item type,
    # but collate_fn transforms (array, int) batches into (Tensor, Tensor) --
    # a shape mypy's stubs don't model, hence the explicit annotation + ignore.
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_sampler=sampler,
        collate_fn=make_collate_fn(embedding_config.backbone),
        num_workers=training_config.dataloader.num_workers,
        pin_memory=training_config.dataloader.pin_memory,
        persistent_workers=training_config.dataloader.persistent_workers,
    )

    model = EmbeddingModel(embedding_config)

    print(f"device: {resolve_device(training_config.device)}")
    start = time.time()
    results = train(model, data_loader, training_config)
    elapsed = time.time() - start

    first_epoch = training_config.num_epochs - len(results)
    for offset, result in enumerate(results):
        print(f"epoch {first_epoch + offset}: mean_loss={result.mean_loss:.4f}")
    print(f"total training time: {elapsed:.1f}s")
    if training_config.checkpoint_dir is not None:
        print(f"final checkpoint: {training_config.checkpoint_dir / 'model.pt'}")
    return results
