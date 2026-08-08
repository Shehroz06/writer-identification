"""One-time data-prep script: converts the Firemaker + CERUG-EN word-crop
archives (downloaded from Zenodo into data/incoming/firemaker_cerug/) into
the imagefolder + metadata.jsonl convention used by this project's real-data
training scripts, under data/raw/firemaker_cerug/{train,test}/.

The archives' own train/test split is per-*sample*, not per-writer (every
writer appears in both halves) -- verified by inspecting the actual writer-ID
overlap before writing this script, not assumed. That's unsuitable for
identification evaluation on genuinely unseen writers, so this script pools
all four archives and re-splits writer-disjoint instead, matching
scripts/prepare_cvl.py and scripts/prepare_cedar.py.

Not part of `handwriting_engine` -- a standalone data-wrangling
utility, per the "freeze the architecture" validation pass.

Usage: uv run python -m scripts.prepare_firemaker
"""

from __future__ import annotations

import json
import random
import re
import tarfile
from pathlib import Path

from PIL import Image

from scripts._common import already_prepared

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "data" / "incoming" / "firemaker_cerug"
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "firemaker_cerug"

_ARCHIVES = [
    ("CERUG-EN-train-images.tar.gz", "cerug", re.compile(r"^train/(Writer\d+)_")),
    ("CERUG-EN-test-images.tar.gz", "cerug", re.compile(r"^test/(Writer\d+)_")),
    ("Firemaker-train-images.tar.gz", "firemaker", re.compile(r"^train/(\d+)-")),
    ("Firemaker-test-images.tar.gz", "firemaker", re.compile(r"^test/(\d+)-")),
]

_TEST_FRACTION = 0.2
_SEED = 42


def _writer_split(writer_ids: list[str]) -> dict[str, str]:
    rng = random.Random(_SEED)
    shuffled = sorted(writer_ids)
    rng.shuffle(shuffled)
    num_test = max(1, int(len(shuffled) * _TEST_FRACTION))
    test_writers = set(shuffled[:num_test])
    return {writer: ("test" if writer in test_writers else "train") for writer in shuffled}


def main() -> None:
    if already_prepared(OUTPUT_DIR):
        print(f"{OUTPUT_DIR} already prepared (train/test metadata.jsonl present) -- skipping.")
        return

    entries: list[tuple[Path, str, str]] = []  # (archive_path, member_name, namespaced_writer_id)
    writer_ids: set[str] = set()

    for archive_name, source_prefix, pattern in _ARCHIVES:
        archive_path = SOURCE_DIR / archive_name
        if not archive_path.exists():
            raise SystemExit(f"Expected archive not found: {archive_path}")
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                match = pattern.match(member.name)
                if not match:
                    continue
                writer_id = f"{source_prefix}_{match.group(1)}"
                entries.append((archive_path, member.name, writer_id))
                writer_ids.add(writer_id)

    if not entries:
        raise SystemExit("No entries matched the expected filename patterns.")

    split_by_writer = _writer_split(sorted(writer_ids))
    rows_by_split: dict[str, list[dict[str, object]]] = {"train": [], "test": []}
    for split in rows_by_split:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {}
    open_archives: dict[Path, tarfile.TarFile] = {}
    try:
        for archive_path, member_name, writer_id in entries:
            if archive_path not in open_archives:
                open_archives[archive_path] = tarfile.open(archive_path)
            archive = open_archives[archive_path]

            split = split_by_writer[writer_id]
            index = counters.get(writer_id, 0)
            counters[writer_id] = index + 1
            file_name = f"{writer_id}_{index}.png"

            extracted = archive.extractfile(member_name)
            if extracted is None:
                continue
            image = Image.open(extracted).convert("L")
            image.save(OUTPUT_DIR / split / file_name)
            rows_by_split[split].append({"file_name": file_name, "writer_id": writer_id})
    finally:
        for archive in open_archives.values():
            archive.close()

    for split, rows in rows_by_split.items():
        metadata_path = OUTPUT_DIR / split / "metadata.jsonl"
        with metadata_path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        num_writers = len({row["writer_id"] for row in rows})
        print(f"{split}: {len(rows)} rows, {num_writers} writers -> {metadata_path}")


if __name__ == "__main__":
    main()
