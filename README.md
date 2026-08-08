# Writer Identification

A child project built on the [Handwriting Analysis Engine](../handwriting-detection-engine) —
ranks a handwriting sample against a gallery of known writers, via a
DINOv2-embedding + Circle-Loss model trained on CVL + Firemaker/CERUG.

This project owns its own dataset, training run, checkpoint, and evaluation.
It does not modify the core engine — it only depends on it.

## Depending on the engine

Right now `pyproject.toml` points at the engine via an absolute local path
(uv/hatchling doesn't reliably resolve a *relative* `file:` reference across
the wheel-metadata parsing step, so this has to be absolute):

```toml
dependencies = ["handwriting-engine @ file:///home/shehroz/Documents/VS%20Code/Projects/Handwriting%20detection"]
```

Update that path if you rename or relocate the engine repo. Once the engine
repo is pushed to GitHub, swap this for:

```toml
dependencies = ["handwriting-engine @ git+https://github.com/Shehroz06/handwriting-detection-engine.git"]
```

## Setup

```bash
uv sync --group dev
uv run pytest   # confirms the install works, no network or checkpoint required
```

## Get the trained checkpoint

Checkpoints aren't committed to git (too large). Place the trained checkpoint at:

```
models/checkpoints/writer_id/
    best_model.pt
    model.pt
    training_state.pt
    labels.json
```

## Use it

Writer identification ranks a query against a gallery — build one first (a
`.npz` of `embeddings` + `labels` arrays; see `scripts/inference.py`'s module
docstring), then:

```bash
uv run python -m writer_identification.backend.cli image.png --gallery gallery.npz
```

Or as a library:

```python
from pathlib import Path
from writer_identification.backend import IdentificationAdapter, IdentificationAppConfig

config = IdentificationAppConfig(
    checkpoint_path=Path("models/checkpoints/writer_id/best_model.pt"),
    gallery_path=Path("gallery.npz"),
)
adapter = IdentificationAdapter.from_config(config)
matches = adapter.identify(image, top_k=5)  # numpy uint8 array in
```

## Train / re-train

Datasets aren't committed either — see `data/README.md` for acquisition + prep:

```bash
uv run python -m scripts.prepare_cvl
uv run python -m scripts.prepare_firemaker
uv run python -m scripts.train corpus=writer_id
uv run python -m scripts.evaluate_writer_id
```

GPU (local CUDA, Kaggle, or Colab): `export UV_NO_SOURCES_PACKAGE="torch torchvision"`
before `uv sync`, then `training.device=cuda` on any command above. See
`scripts/colab_bootstrap.py` for Colab Drive-persistence helpers.

## Project structure

```
src/writer_identification/backend/    adapter, config, CLI
corpora/                               CVL + Firemaker/CERUG loaders, plus IAM/IAM-HistDB
                                        (not currently used by the training pipeline --
                                        kept as reference for future corpus additions)
scripts/                               train / evaluate / prepare / inference glue
configs/                               Hydra configuration
tests/
models/checkpoints/                    gitignored -- see "Get the trained checkpoint"
data/                                  gitignored -- see data/README.md
```
