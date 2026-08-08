"""One-time-per-Colab-session bootstrap: mounts Google Drive and symlinks
this repo's `data/raw/`, `data/incoming/`, and `models/checkpoints/`
directories to a persistent folder on Drive, so downloaded datasets and
training checkpoints survive a Colab runtime disconnect instead of living
only in ephemeral `/content` storage.

If a directory already has real (non-symlink) local data -- e.g. this
session downloaded/prepared datasets, or trained a checkpoint, before
running this script -- and Drive doesn't have anything there yet, that
local data is copied *up* to Drive first, then replaced with a symlink.
Note that **Drive storage belongs to whichever Google account is logged
in**: switching accounts (e.g. to dodge Colab's per-account session limit)
means a completely different, empty Drive -- this script cannot carry a
checkpoint across accounts, only within repeated sessions of the *same*
account. For cross-account (or Colab<->Kaggle) continuity, download the
checkpoint to your own machine at the end of a session and re-upload it as
`training.init_checkpoint` at the start of the next one, regardless of
which account/platform -- see README's "Moving checkpoints between
machines".

Deliberately just filesystem plumbing -- no engine or training code changes
anywhere. Once the symlinks exist, `scripts/train_*.py`'s default
`checkpoint_dir` and `scripts/prepare_*.py`'s default `data/raw` output
already point through them transparently: checkpoints land directly on
Drive every epoch (already the training loop's existing behavior), and
resuming automatically (`training.resume=true`, the default) picks up
Drive-persisted checkpoints exactly like local ones, since a mounted Drive
folder is just another directory to `pathlib`/`torch.save`.

**Colab detection note**: this script is meant to be launched with
`uv run python -m scripts.colab_bootstrap`, which runs in *this project's
own* `uv`-managed virtual environment -- a separate Python installation
from the Colab notebook kernel's system Python, where `google.colab` is
preinstalled. `import google.colab` therefore correctly fails inside a `uv
run` subprocess even in a genuine Colab session, so detection here falls
back to environment variables Colab sets at the container/VM level (visible
to every process in that VM, including `uv run` subprocesses, since they're
inherited via `os.environ` regardless of which Python venv runs) and,
failing that, the filesystem path Colab's system Python actually installs
`google.colab` under. See `_running_in_colab`.

Similarly, *mounting* Drive requires calling `google.colab.drive.mount()`,
which only works from the notebook kernel's own process (it drives an OAuth
handshake with the notebook frontend) -- a `uv run` subprocess cannot
perform that mount no matter how well it detects Colab. If Drive isn't
already mounted, this script says so and tells you to mount it from a
notebook cell first, rather than silently doing nothing or pretending it
succeeded. See `_mount_drive`.

Safe to re-run every session (idempotent): an existing correct symlink is
left alone; a real (non-symlink) directory already present locally is left
alone too, with a clear error rather than silently overwriting or orphaning
whatever's in it. Running outside Colab is a no-op with a clear message, so
this can be called unconditionally from a notebook cell.

Not part of `handwriting_engine` -- standalone Colab-session glue.

Usage:
    # In a notebook cell, mount Drive first (only `google.colab` itself,
    # running in the notebook kernel, can do this):
    from google.colab import drive
    drive.mount("/content/drive")

    # Then, in a notebook cell or terminal:
    !uv run python -m scripts.colab_bootstrap
    # or with a custom Drive folder name (e.g. to keep multiple projects separate):
    !uv run python -m scripts.colab_bootstrap --drive-dir /content/drive/MyDrive/my-project
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVE_DIR = Path("/content/drive/MyDrive/handwriting-detection")
_LINKED_DIRS = ("data/raw", "data/incoming", "models/checkpoints")

_DRIVE_MOUNT_ROOT = Path("/content/drive")

# Env vars Colab sets at the container level (present for every process in
# the VM, regardless of which Python venv runs it) -- checked in addition to
# `import google.colab` since that import only succeeds in Colab's own
# system Python, not a `uv run`-managed virtual environment. Several are
# checked, not just one, since which exact vars a given Colab runtime image
# sets has varied over time and isn't part of any stable public contract.
_COLAB_ENV_MARKERS = (
    "COLAB_RELEASE_TAG",
    "COLAB_GPU",
    "COLAB_BACKEND_VERSION",
    "COLAB_JUPYTER_TRANSPORT_PROTOCOL",
)

# Colab's system Python installs `google.colab` here (Debian/Ubuntu-style
# `dist-packages`, not the more common `site-packages` a plain `pip install`
# venv would use) -- checked as a filesystem path, not an import, so it's
# visible from a *different* Python venv running in the same VM.
_COLAB_SITE_PACKAGES_GLOBS = (
    "/usr/local/lib/python3*/dist-packages/google/colab",
    "/usr/lib/python3*/dist-packages/google/colab",
)


def _running_in_colab() -> bool:
    """Detects Google Colab robustly, even when this process is a `uv
    run`-managed virtual environment separate from the notebook kernel
    (where `import google.colab` would normally succeed). See the module
    docstring for why a single `sys.modules`/import check isn't reliable
    here.
    """
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        pass

    if any(marker in os.environ for marker in _COLAB_ENV_MARKERS):
        return True

    return any(
        candidate_path
        for pattern in _COLAB_SITE_PACKAGES_GLOBS
        for candidate_path in Path("/").glob(pattern.lstrip("/"))
    )


def _drive_already_mounted() -> bool:
    """Whether Drive is already mounted at `/content/drive` -- checked via
    the filesystem (a mounted Drive is a FUSE mount visible VM-wide) rather
    than by trying to import `google.colab`, since a `uv run` subprocess can
    use an already-mounted Drive just fine even though it can't perform the
    mount itself.
    """
    return (_DRIVE_MOUNT_ROOT / "MyDrive").is_dir()


def _mount_drive() -> bool:
    """Ensures Google Drive is mounted at `/content/drive`, returning
    whether it's usable.

    1. Already mounted (e.g. from an earlier notebook cell) -- use it as-is.
    2. Not mounted, but `google.colab` is importable in *this* process
       (e.g. this script was run directly in the notebook kernel rather
       than via `uv run`) -- mount it now.
    3. Not mounted, and `google.colab` isn't importable here (the expected
       case when launched via `uv run`) -- this process cannot perform the
       OAuth mount handshake; say so clearly instead of silently doing
       nothing.
    """
    if _drive_already_mounted():
        return True

    if not _running_in_colab():
        print("Not running in Google Colab -- skipping Drive mount and symlink setup.")
        return False

    try:
        from google.colab import drive
    except ImportError:
        print(
            "Detected Google Colab, but Google Drive isn't mounted yet, and this "
            "process (running via `uv run`, in this project's own virtual "
            "environment) can't perform the mount itself -- `google.colab` only "
            "exists in Colab's system Python, not here. Mount Drive from a "
            "notebook cell first, then re-run this script:\n"
            "    from google.colab import drive\n"
            '    drive.mount("/content/drive")'
        )
        return False

    drive.mount(str(_DRIVE_MOUNT_ROOT), force_remount=False)
    return True


def _link(relative_path: str, drive_dir: Path) -> bool:
    """Symlinks one directory, returning whether it ended up linked.

    Each of `_LINKED_DIRS` is independent -- e.g. `data/raw` already having
    real (non-symlink) data from earlier in the same session must not stop
    `models/checkpoints` from still being linked. Returning a bool rather
    than raising lets `main` attempt every directory regardless of earlier
    ones' outcomes, instead of a `SystemExit` on the first conflict silently
    aborting the whole run short of directories that had no conflict at all.

    If `local_path` is already a real (non-symlink) directory -- e.g. this
    session prepared data before bootstrapping, or trained a checkpoint
    before Drive was mounted -- and the Drive side is still empty, the local
    data is copied *up* to Drive first, then replaced with a symlink, so
    whichever session/account happens to have the freshest local data is
    the one that populates Drive; nothing is deleted until the copy to
    Drive has fully succeeded. If Drive already has content too, this is a
    genuine conflict this function won't guess how to resolve (blindly
    merging, e.g. two different checkpoints, would be actively wrong) --
    reported clearly instead.
    """
    local_path = REPO_ROOT / relative_path
    drive_path = drive_dir / relative_path
    drive_path.mkdir(parents=True, exist_ok=True)

    if local_path.is_symlink():
        if local_path.resolve() == drive_path.resolve():
            print(f"{relative_path}: already linked to Drive.")
            return True
        local_path.unlink()
    elif local_path.exists():
        if any(drive_path.iterdir()):
            print(
                f"{relative_path}: skipped -- both {local_path} (local) and "
                f"{drive_path} (Drive) already have content; won't guess which "
                "should win. Reconcile manually, e.g. copy anything Drive is "
                f"missing without touching what's already there: `cp -rn "
                f"{local_path}/. {drive_path}/`, remove {local_path}, then "
                "re-run this script."
            )
            return False
        print(f"{relative_path}: copying existing local data to Drive (first time seen there)...")
        shutil.copytree(local_path, drive_path, dirs_exist_ok=True)
        shutil.rmtree(local_path)
        print(f"{relative_path}: copied to {drive_path}.")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.symlink_to(drive_path, target_is_directory=True)
    print(f"{relative_path} -> {drive_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=DEFAULT_DRIVE_DIR,
        help=f"Persistent Drive folder to link against (default: {DEFAULT_DRIVE_DIR}).",
    )
    args = parser.parse_args()

    if not _mount_drive():
        return

    linked = [
        relative_path for relative_path in _LINKED_DIRS if _link(relative_path, args.drive_dir)
    ]
    skipped = [relative_path for relative_path in _LINKED_DIRS if relative_path not in linked]
    if skipped:
        print(
            f"\n{len(linked)}/{len(_LINKED_DIRS)} directories linked; skipped: "
            f"{', '.join(skipped)} (see messages above for why -- each directory is "
            "independent, so this doesn't affect the ones that did link)."
        )
    if not linked:
        return

    print(
        f"\nBootstrap complete for {', '.join(linked)} -- these now "
        f"live under {args.drive_dir} on Drive -- downloaded/prepared datasets and "
        "training checkpoints persist across Colab sessions automatically. No changes "
        "needed to any training command: scripts/train_*.py's default checkpoint_dir "
        "and scripts/prepare_*.py's default output already read/write through these "
        "paths, and training.resume=true (the default) picks up a Drive-persisted "
        "checkpoint exactly like a local one."
    )


if __name__ == "__main__":
    main()
