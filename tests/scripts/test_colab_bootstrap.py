"""Tests for scripts.colab_bootstrap's Colab-detection and Drive-mount
logic.

Exists (breaking this project's usual convention of verifying `scripts/`
via direct smoke-test execution rather than pytest) specifically because
this logic can't be smoke-tested against real Colab from this environment
at all -- these tests are the only verification available, so they're
worth the exception.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from scripts import colab_bootstrap


@pytest.fixture(autouse=True)
def _clean_colab_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from "definitely not Colab": no marker env vars,
    no injected `google.colab` module, on this reference machine's real
    (non-Colab) filesystem."""
    for marker in colab_bootstrap._COLAB_ENV_MARKERS:
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google", raising=False)


def test_running_in_colab_false_with_no_signals() -> None:
    assert colab_bootstrap._running_in_colab() is False


def test_running_in_colab_true_via_importable_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """The case that works when this script runs directly in the notebook
    kernel (not via `uv run`)."""
    fake_google = types.ModuleType("google")
    fake_colab = types.ModuleType("google.colab")
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)

    assert colab_bootstrap._running_in_colab() is True


@pytest.mark.parametrize("marker", colab_bootstrap._COLAB_ENV_MARKERS)
def test_running_in_colab_true_via_env_var_marker(
    marker: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case that matters for `uv run python -m scripts.colab_bootstrap`
    -- `google.colab` genuinely isn't importable in that separate venv, so
    detection must fall back to container-level env vars."""
    monkeypatch.setenv(marker, "1")
    assert colab_bootstrap._running_in_colab() is True


def test_running_in_colab_true_via_filesystem_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Last-resort fallback: Colab's system Python's dist-packages, checked
    as a path (not an import) so it's visible even from a different venv."""
    fake_site_packages = tmp_path / "python3.11" / "dist-packages" / "google" / "colab"
    fake_site_packages.mkdir(parents=True)
    monkeypatch.setattr(
        colab_bootstrap,
        "_COLAB_SITE_PACKAGES_GLOBS",
        (str(tmp_path / "python3*" / "dist-packages" / "google" / "colab"),),
    )

    assert colab_bootstrap._running_in_colab() is True


def test_drive_already_mounted_false_when_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(colab_bootstrap, "_DRIVE_MOUNT_ROOT", tmp_path / "drive")
    assert colab_bootstrap._drive_already_mounted() is False


def test_drive_already_mounted_true_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    drive_root = tmp_path / "drive"
    (drive_root / "MyDrive").mkdir(parents=True)
    monkeypatch.setattr(colab_bootstrap, "_DRIVE_MOUNT_ROOT", drive_root)

    assert colab_bootstrap._drive_already_mounted() is True


def test_mount_drive_uses_already_mounted_drive_without_importing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    drive_root = tmp_path / "drive"
    (drive_root / "MyDrive").mkdir(parents=True)
    monkeypatch.setattr(colab_bootstrap, "_DRIVE_MOUNT_ROOT", drive_root)

    assert colab_bootstrap._mount_drive() is True


def test_mount_drive_returns_false_outside_colab(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(colab_bootstrap, "_DRIVE_MOUNT_ROOT", tmp_path / "drive")
    assert colab_bootstrap._mount_drive() is False


def test_mount_drive_calls_drive_mount_when_module_importable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Detected via env var (simulating a `uv run` subprocess correctly
    identifying Colab), with `google.colab` *also* importable here (e.g.
    this script run directly in the notebook kernel) -- should actually
    call `drive.mount`."""
    monkeypatch.setattr(colab_bootstrap, "_DRIVE_MOUNT_ROOT", tmp_path / "drive")
    monkeypatch.setenv(colab_bootstrap._COLAB_ENV_MARKERS[0], "1")

    mount_calls: list[tuple[str, bool]] = []
    fake_drive = types.ModuleType("google.colab.drive")
    fake_drive.mount = lambda path, force_remount=False: mount_calls.append(  # type: ignore[attr-defined]
        (path, force_remount)
    )
    fake_colab = types.ModuleType("google.colab")
    fake_colab.drive = fake_drive  # type: ignore[attr-defined]
    fake_google = types.ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    monkeypatch.setitem(sys.modules, "google.colab.drive", fake_drive)

    assert colab_bootstrap._mount_drive() is True
    assert mount_calls == [(str(tmp_path / "drive"), False)]


def test_mount_drive_returns_false_when_detected_but_not_importable_here(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The actual bug being fixed: detected via env var (a `uv run`
    subprocess correctly identifying Colab), but `google.colab` genuinely
    isn't importable in this venv -- must report this clearly rather than
    crash or silently pretend it mounted."""
    monkeypatch.setattr(colab_bootstrap, "_DRIVE_MOUNT_ROOT", tmp_path / "drive")
    monkeypatch.setenv(colab_bootstrap._COLAB_ENV_MARKERS[0], "1")

    assert colab_bootstrap._mount_drive() is False


def test_link_creates_symlink_and_returns_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(colab_bootstrap, "REPO_ROOT", repo_root)
    drive_dir = tmp_path / "drive_project"

    assert colab_bootstrap._link("data/raw", drive_dir) is True
    local_path = repo_root / "data" / "raw"
    assert local_path.is_symlink()
    assert local_path.resolve() == (drive_dir / "data" / "raw").resolve()


def test_link_migrates_local_data_to_empty_drive_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common case for a fresh account/session: local has real prepared
    data, Drive doesn't have anything there yet -- the local data should be
    copied up to Drive, then replaced with a symlink, not just skipped."""
    repo_root = tmp_path / "repo"
    local_raw = repo_root / "data" / "raw"
    local_raw.mkdir(parents=True)
    (local_raw / "cvl" / "train").mkdir(parents=True)
    (local_raw / "cvl" / "train" / "metadata.jsonl").write_text('{"writer_id": "cvl_1"}')
    monkeypatch.setattr(colab_bootstrap, "REPO_ROOT", repo_root)
    drive_dir = tmp_path / "drive_project"

    assert colab_bootstrap._link("data/raw", drive_dir) is True

    assert local_raw.is_symlink()
    migrated = drive_dir / "data" / "raw" / "cvl" / "train" / "metadata.jsonl"
    assert migrated.read_text() == '{"writer_id": "cvl_1"}'


def test_link_returns_false_when_both_sides_already_have_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine conflict -- both local and Drive already have (possibly
    different) content -- must be left alone and reported, not guessed at
    or silently overwritten in either direction."""
    repo_root = tmp_path / "repo"
    local_raw = repo_root / "data" / "raw"
    local_raw.mkdir(parents=True)
    (local_raw / "local_marker.txt").write_text("local data")
    monkeypatch.setattr(colab_bootstrap, "REPO_ROOT", repo_root)

    drive_dir = tmp_path / "drive_project"
    drive_raw = drive_dir / "data" / "raw"
    drive_raw.mkdir(parents=True)
    (drive_raw / "drive_marker.txt").write_text("drive data")

    assert colab_bootstrap._link("data/raw", drive_dir) is False
    # Untouched on both sides: still real directories, not replaced or merged.
    assert not local_raw.is_symlink()
    assert (local_raw / "local_marker.txt").exists()
    assert (drive_raw / "drive_marker.txt").exists()


def test_main_links_remaining_directories_despite_one_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the actual reported bug: data/raw already
    conflicting on both sides (e.g. prepared earlier in the same session,
    with unrelated content already on Drive too) must not prevent
    models/checkpoints from still being linked -- previously, `_link`
    raising `SystemExit` on the first conflict aborted `main`'s loop
    entirely, silently skipping every directory after it.
    """
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "raw").mkdir(parents=True)
    (repo_root / "data" / "raw" / "local_marker.txt").write_text("local data")
    monkeypatch.setattr(colab_bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(colab_bootstrap, "_mount_drive", lambda: True)

    drive_dir = tmp_path / "drive_project"
    (drive_dir / "data" / "raw").mkdir(parents=True)
    (drive_dir / "data" / "raw" / "drive_marker.txt").write_text("drive data")
    checkpoint_source = drive_dir / "models" / "checkpoints" / "writer_id"
    checkpoint_source.mkdir(parents=True)
    (checkpoint_source / "best_model.pt").write_text("fake checkpoint")

    monkeypatch.setattr(sys, "argv", ["colab_bootstrap", "--drive-dir", str(drive_dir)])
    colab_bootstrap.main()

    assert not (repo_root / "data" / "raw").is_symlink()  # left alone, as expected
    linked_checkpoint = repo_root / "models" / "checkpoints" / "writer_id" / "best_model.pt"
    assert linked_checkpoint.is_symlink() is False  # it's a real file behind a linked dir
    assert linked_checkpoint.exists()  # the actual regression: this used to be unreachable
