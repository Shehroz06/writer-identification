# Results

Findings that only ever lived in `outputs/` (gitignored, Hydra per-run logs)
or terminal scrollback get recorded here so they survive.

## Corpus linear-probe / cross-corpus separability diagnostic

Run: `uv run python -m scripts.probe_corpus_separability`, 2026-08-11
(`outputs/2026-08-11/11-12-05/probe_corpus_separability.log`).

Held-out test corpus: 10,489 images, 133 writers, 133 queries, 10,356 gallery entries.

| Probe | Balanced accuracy | ROC-AUC |
|---|---|---|
| Baseline (frozen/untrained DINOv2) | 1.0000 | 1.0000 |
| Trained (`model.pt`) | 1.0000 | 1.0000 |

Both the untrained and trained backbone perfectly separate CVL vs.
Firemaker/CERUG by corpus alone — i.e. embeddings carry a strong corpus
signal independent of any writer-ID training. This motivated the
cross-corpus similarity diagnostic below, to check whether that corpus
signal is actually confounding writer identification.

**Cross-corpus similarity diagnostic** (same run, trained model):

| Comparison | Mean cosine similarity | Median | n |
|---|---|---|---|
| Genuine match (same writer) | 0.8529 | 0.8573 | 133 |
| Same-corpus foil (different writer, same corpus) | 0.8349 | 0.8425 | 133 |
| Cross-corpus foil (different writer, different corpus) | 0.6273 | 0.6308 | 133 |

Mann-Whitney U (same-corpus foil > cross-corpus foil): statistic=17656.0,
p=4.16e-45 — the corpus signal is real and highly significant, but genuine
matches still score clearly higher than same-corpus foils (0.8529 vs.
0.8349), so writer identity is not simply being decided by corpus
membership; the corpus signal is a confound sitting *underneath* a real,
separable writer signal, not a substitute for it.

## Held-out writer-ID identification benchmark

Top-1 accuracy ~62% (Circle-Loss fine-tuned embeddings vs. untrained-DINOv2
baseline) was reported from an interactive run of
`scripts/evaluate_writer_id.py` in an earlier session, but the run predated
this script's `print()` → `logger.info()` fix, so Hydra's per-run log never
captured it and it was never written to a file. Not yet reproduced/saved.

That 62% baseline was trained with the backbone **100% frozen**
(`unfreeze_last_blocks: 0`) and **zero data augmentation** (the
`configs/preprocessing/augmentation.yaml` config existed but was never
actually wired into the training path — confirmed by reading
`scripts/_common.py`'s training path end-to-end, no reference to it
anywhere) for only 10 epochs x 100 batches. Only the 128-dim embedding head
was ever trained; nothing ever adapted to handwriting-specific style, and
the corpus-confound found above (embeddings trivially separate CVL vs.
Firemaker/CERUG) had nothing pushing against it.

**Changes made to raise this** (uncommitted, pending a training run):
- `configs/embeddings/default.yaml`: `unfreeze_last_blocks: 4` — Stage 2
  fine-tuning of the backbone's last 4 transformer blocks, at a separate
  (lower, `1e-5` default) learning rate from the head.
- `configs/config.yaml` / `scripts/_common.py`: augmentation is now actually
  applied during training (rotation/elastic/grid/optical distortion,
  brightness/contrast/noise), which should also erode the corpus-confound
  by making scan-condition artifacts a less reliable shortcut.
- `configs/training/default.yaml`: 30 epochs x 200 batches/epoch (was 10x100),
  warmup+cosine-decay scheduler enabled, AMP enabled (GPU-only, no-op on CPU).
- `pyproject.toml`: added `datasets` as an explicit dependency — it was a
  real, direct import in `corpora/*.py` that had never actually been
  declared, only working locally by chance; this silently blocked
  `scripts.train`/`scripts.prepare_*` (and would have blocked the Colab run)
  until fixed.

Re-run `scripts.evaluate_writer_id` against the new checkpoint and record
top-1/mAP here once training completes.
