"""One-time data-prep script: converts the CVL Database cropped release
(downloaded from Zenodo into data/incoming/cvl-database-cropped-1-1.zip) into
the imagefolder + metadata.jsonl convention
`corpora.cvl.load_cvl` expects, under
data/raw/cvl/{train,test}/ -- fulfilling the manual-conversion contract
documented in data/README.md since Phase 4.

Not part of `handwriting_engine` -- a standalone data-wrangling
utility, per the "freeze the architecture" validation pass.

Usage: uv run python -m scripts.prepare_cvl
"""

from __future__ import annotations

import json
import random
import re
import zipfile
from pathlib import Path

from PIL import Image

from scripts._common import already_prepared

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = REPO_ROOT / "data" / "incoming" / "cvl-database-cropped-1-1.zip"
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "cvl"

_ENTRY_PATTERN = re.compile(r"^cvl-database-cropped-1-1/(\d{4})-(\d+)-cropped\.tif$")

_TEST_FRACTION = 0.2
_SEED = 42


def _writer_split(writer_ids: list[str]) -> dict[str, str]:
    """Assign each writer to train/test, writer-disjoint, so identification
    evaluation is on genuinely unseen writers."""
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

    if not SOURCE_ZIP.exists():
        raise SystemExit(f"CVL archive not found at {SOURCE_ZIP}")

    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        entries: list[tuple[str, str, str]] = []
        writer_ids: set[str] = set()
        for name in archive.namelist():
            match = _ENTRY_PATTERN.match(name)
            if match:
                writer, page = match.groups()
                entries.append((name, writer, page))
                writer_ids.add(writer)

        if not entries:
            raise SystemExit("No CVL entries matched the expected filename pattern.")

        split_by_writer = _writer_split(sorted(writer_ids))
        rows_by_split: dict[str, list[dict[str, object]]] = {"train": [], "test": []}
        for split in rows_by_split:
            (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

        for name, writer, page in entries:
            split = split_by_writer[writer]
            file_name = f"cvl_{writer}_{page}.png"
            with archive.open(name) as fh:
                image = Image.open(fh).convert("L")
                image.save(OUTPUT_DIR / split / file_name)
            rows_by_split[split].append({"file_name": file_name, "writer_id": f"cvl_{writer}"})

    for split, rows in rows_by_split.items():
        metadata_path = OUTPUT_DIR / split / "metadata.jsonl"
        with metadata_path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        num_writers = len({row["writer_id"] for row in rows})
        print(f"{split}: {len(rows)} rows, {num_writers} writers -> {metadata_path}")


if __name__ == "__main__":
    main()
