# Data directory

Everything under `data/` (other than this file) is gitignored -- nothing is
committed or downloaded automatically.

## CVL Database (310 writers)

Not on the HuggingFace Hub (TU Wien, CC BY-NC 3.0). Download
`cvl-database-cropped-1-1.zip` from `zenodo.org/records/1492267` into
`data/incoming/`, then:

```bash
uv run python -m scripts.prepare_cvl
```

## Firemaker + CERUG-EN (355 writers)

Download all four archives from `zenodo.org/records/13258163` into
`data/incoming/firemaker_cerug/`, then:

```bash
uv run python -m scripts.prepare_firemaker
```

Both re-split writer-disjoint (80/20, seed 42) into:

```
data/raw/<cvl|firemaker_cerug>/
  train/
    metadata.jsonl   # {"file_name": ..., "writer_id": "<source>_<n>"}
    <images>
  test/
    metadata.jsonl
    <images>
```
