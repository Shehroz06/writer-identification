# Writer Identification

A downstream project built on the [Handwriting Analysis Engine](https://github.com/Shehroz06/handwriting-detection-engine) —
ranks a handwriting sample against a gallery of known writers, via a
DINOv2-embedding + Circle-Loss model trained on CVL + Firemaker/CERUG.

This project owns its own dataset, training run, checkpoint, and evaluation.
It does not modify the core engine — it only depends on it.

## Depending on the engine

`pyproject.toml` points at the engine's public GitHub repo over plain HTTPS,
pinned to a tag for reproducible builds:

```toml
dependencies = ["handwriting-engine @ git+https://github.com/Shehroz06/handwriting-detection-engine.git@v0.1.0"]
```

No credentials or SSH setup needed — `uv sync` (below) fetches it like any
other dependency.

## Setup

```bash
uv sync --group dev
uv run pytest   # confirms the install works, no network or checkpoint required
```

## Get the trained checkpoint

Checkpoints aren't committed to git (too large). Download the trained
checkpoint from
[Google Drive](https://drive.google.com/drive/folders/1QtA1619sCCNgFCDzmtkeHslmVcE9lQ0T?usp=sharing)
and place it at:

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

## Run the service

A local/internal HTTP service wrapping the same `IdentificationAdapter` used
above — no auth/rate-limiting/public hosting (not needed for local/internal
use), but a `Depends()`-based auth dependency could be added to
`src/writer_identification/backend/api/routes.py` later without restructuring
anything else.

**Input contract**: both `/identify` and `/enroll` expect an already-cropped
single word/line handwriting sample, matching exactly what CVL and
Firemaker/CERUG contain — no perspective correction, denoising, or
segmentation happens anywhere in this pipeline. A full page photo gets
embedded as if it were one giant "word," producing a meaningless result, not
just a less accurate one.

### Directly via uvicorn

```bash
uv run uvicorn writer_identification.backend.api.app:app --port 8000
```

Reads the checkpoint/gallery from `models/checkpoints/writer_id/` by default,
overridable via `WRITER_ID_CHECKPOINT_PATH` / `WRITER_ID_GALLERY_PATH`.

```bash
curl http://localhost:8000/health

curl -F "file=@data/raw/cvl/test/cvl_0216_3.png" "http://localhost:8000/identify?top_k=3"

curl -F "writer_id=alice" -F "files=@me1.png" -F "files=@me2.png" http://localhost:8000/enroll
```

### Via Docker

The checkpoint/gallery are volume-mounted, not baked into the image, so
swapping in a retrained model or rebuilt gallery never requires a rebuild:

```bash
docker build -t writer-identification-api .
docker run -d -p 8000:8000 \
  -v "$(pwd)/models/checkpoints/writer_id:/models/writer_id:ro" \
  --name wid writer-identification-api
```

The example mount above is read-only, which is fine for `/health` and
`/identify` but means `/enroll` can't persist — drop `:ro` (or mount a
writable copy) if you want enrollment to work through the containerized
service.

```bash
curl http://localhost:8000/health
curl -F "file=@data/raw/cvl/test/cvl_0216_3.png" "http://localhost:8000/identify?top_k=3"
docker logs wid   # confirm no startup error about a missing checkpoint/gallery
docker stop wid && docker rm wid
```

### Enroll writers offline (no server needed)

```bash
uv run python -m scripts.enroll_writer --writer-id alice path/to/sample1.png path/to/sample2.png
```

Both enrollment paths mutate `models/checkpoints/writer_id/gallery.npz` in
place — back it up first (`cp gallery.npz gallery.npz.bak`) if you want the
original benchmark gallery recoverable later. This doesn't affect the
held-out accuracy numbers already reported (evaluation uses
`data/raw/*/test/` directly, never `gallery.npz`).

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

### Stage 2 fine-tuning (current default)

`configs/embeddings/default.yaml` unfreezes the backbone's last 4 transformer
blocks (`unfreeze_last_blocks: 4`) instead of training a head on 100% frozen
DINOv2 features, `configs/config.yaml` wires up
`configs/preprocessing/augmentation.yaml`'s stochastic augmentation
(rotation/elastic/grid/optical distortion, brightness/contrast/noise) into
the training path for the first time, and `configs/training/default.yaml`
bumps the budget to 30 epochs x 200 batches/epoch with a warmup+cosine-decay
scheduler and AMP (the 62% top-1 baseline this improves on was trained with
the backbone 100% frozen and zero augmentation). Sized for a GPU run;
override downward for CPU dev iteration, e.g.
`training.num_epochs=3 training.sampler.batches_per_epoch=20`.

After training finishes, replace the local `models/checkpoints/writer_id/`
with the run's output (`best_model.pt`/`model.pt`/`training_state.pt`/
`labels.json`) before rebuilding the gallery and re-evaluating:

```bash
uv run python -m scripts.build_gallery spot_check=true   # cheap sanity check first
uv run python -m scripts.build_gallery                   # full gallery rebuild
uv run python -m scripts.evaluate_writer_id
```

## Project structure

```
src/writer_identification/backend/    adapter, config, CLI
src/writer_identification/backend/api/ FastAPI service (app/routes/schemas/state/config)
corpora/                               CVL + Firemaker/CERUG loaders
scripts/                               train / evaluate / prepare / inference / enroll glue
configs/                               Hydra configuration
tests/
Dockerfile                             API service container (see "Run the service")
models/checkpoints/                    gitignored -- see "Get the trained checkpoint"
data/                                  gitignored -- see data/README.md
```
