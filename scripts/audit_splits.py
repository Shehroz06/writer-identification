"""Independent integrity audit for the writer-disjoint train/test splits baked
into `data/raw/{cvl,firemaker_cerug}/{train,test}/metadata.jsonl`.

Re-derives each corpus's writer->split assignment from the on-disk writer-ID
set alone (no re-download/re-extraction of source archives needed, since
`_writer_split()` is a pure, seeded function of a writer-ID list) and checks
it against what's actually on disk, plus checks for cross-split writer
leakage and cross-corpus writer-ID collisions.

Usage: uv run python -m scripts.audit_splits
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_cvl import _writer_split as _cvl_writer_split
from scripts.prepare_firemaker import _writer_split as _firemaker_writer_split

REPO_ROOT = Path(__file__).resolve().parents[1]
CVL_DIR = REPO_ROOT / "data" / "raw" / "cvl"
FIREMAKER_CERUG_DIR = REPO_ROOT / "data" / "raw" / "firemaker_cerug"


def _load_writer_ids(metadata_path: Path) -> set[str]:
    with metadata_path.open() as fh:
        return {json.loads(line)["writer_id"] for line in fh}


def audit_cvl() -> set[str]:
    train_ids = _load_writer_ids(CVL_DIR / "train" / "metadata.jsonl")
    test_ids = _load_writer_ids(CVL_DIR / "test" / "metadata.jsonl")
    leaked = train_ids & test_ids
    assert not leaked, f"cvl: cross-split writer leakage: {sorted(leaked)}"

    # prepare_cvl._writer_split() is called on *unprefixed* writer IDs
    # (e.g. "0001"); the "cvl_" namespace prefix is appended afterward.
    all_ids = train_ids | test_ids
    raw_ids = sorted(writer_id.removeprefix("cvl_") for writer_id in all_ids)
    recomputed = _cvl_writer_split(raw_ids)

    mismatches = []
    for raw_id, split in recomputed.items():
        namespaced = f"cvl_{raw_id}"
        on_disk = "test" if namespaced in test_ids else "train"
        if split != on_disk:
            mismatches.append((namespaced, split, on_disk))
    assert not mismatches, f"cvl: split-recomputation mismatches: {mismatches}"

    print(
        f"cvl: {len(train_ids)} train writers, {len(test_ids)} test writers, "
        f"0 mismatches, 0 cross-split leakage"
    )
    return all_ids


def audit_firemaker_cerug() -> set[str]:
    train_ids = _load_writer_ids(FIREMAKER_CERUG_DIR / "train" / "metadata.jsonl")
    test_ids = _load_writer_ids(FIREMAKER_CERUG_DIR / "test" / "metadata.jsonl")
    leaked = train_ids & test_ids
    assert not leaked, f"firemaker_cerug: cross-split writer leakage: {sorted(leaked)}"

    # prepare_firemaker._writer_split() is called on already-namespaced
    # writer IDs ("cerug_Writer0101", "firemaker_0042") directly.
    all_ids = train_ids | test_ids
    recomputed = _firemaker_writer_split(sorted(all_ids))

    mismatches = []
    for writer_id, split in recomputed.items():
        on_disk = "test" if writer_id in test_ids else "train"
        if split != on_disk:
            mismatches.append((writer_id, split, on_disk))
    assert not mismatches, f"firemaker_cerug: split-recomputation mismatches: {mismatches}"

    print(
        f"firemaker_cerug: {len(train_ids)} train writers, {len(test_ids)} test writers, "
        f"0 mismatches, 0 cross-split leakage"
    )
    return all_ids


def audit_no_cross_corpus_collision(cvl_ids: set[str], firemaker_cerug_ids: set[str]) -> None:
    collision = cvl_ids & firemaker_cerug_ids
    assert not collision, f"cross-corpus writer-ID collision: {sorted(collision)}"
    for writer_id in cvl_ids:
        assert writer_id.startswith("cvl_"), f"unexpected non-cvl-prefixed ID in CVL: {writer_id}"
    for writer_id in firemaker_cerug_ids:
        assert writer_id.startswith(("cerug_", "firemaker_")), (
            f"unexpected prefix in firemaker_cerug: {writer_id}"
        )
    print("cross-corpus: 0 writer-ID collisions, prefix namespacing verified")


def main() -> None:
    cvl_ids = audit_cvl()
    firemaker_cerug_ids = audit_firemaker_cerug()
    audit_no_cross_corpus_collision(cvl_ids, firemaker_cerug_ids)
    print("\nAudit passed: splits are deterministic, writer-disjoint, and corpus-disjoint.")


if __name__ == "__main__":
    main()
