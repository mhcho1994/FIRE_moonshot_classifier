#!/usr/bin/env python3
"""
Synchronize MOTIF FlightTest data between Windows, WSL, and OneDrive.

This script provides a unified interface for transferring FlightTest datasets
between a Windows filesystem mounted in WSL, a local WSL workspace, and a
OneDrive remote configured through rclone.

Supported operations
--------------------
pull
    Copy data from Windows to WSL.

        Windows D: -> WSL

    Use ``--windows-dir`` to select the Windows source directory.

push
    Copy data from WSL to Windows.

        WSL -> Windows D:

    Use ``--windows-dir`` to select the Windows destination directory.

download
    Download data from OneDrive to WSL.

        OneDrive -> WSL

    The OneDrive source must be specified with ``--onedrive-dir`` using an
    rclone remote path, for example::

        motif_onedrive:MOTIF/FlightTest

upload
    Upload data from WSL to OneDrive.

        WSL -> OneDrive

    The OneDrive destination must be specified with ``--onedrive-dir``.

WSL directory
-------------
The WSL FlightTest directory is specified with ``--wsl-dir``.

The default is::

    ./data/flight_test

The default and any relative ``--wsl-dir`` value are resolved against the
current working directory.

Safety
------
All operations run in dry-run mode by default. The planned file transfers are
displayed without transferring or deleting files. A missing local
destination directory may still be created.

Use ``--execute`` to perform the actual transfer.

Use ``--delete`` to mirror the source to the destination. When enabled, files
that exist only in the destination may be deleted. This option should therefore
be used with caution.

Transfer backends
-----------------
Windows <-> WSL transfers use ``rsync``.

OneDrive <-> WSL transfers use ``rclone``.

Exclusion options
-----------------
No exclusion option
    Include every file.

--default-excludes
    Apply DEFAULT_RSYNC_EXCLUDES for pull/push or DEFAULT_RCLONE_EXCLUDES for
    download/upload.

--exclude PATTERN
    Add a custom exclusion; repeat the option to add more patterns.

--default-excludes with --exclude PATTERN
    Apply the backend's defaults and every custom pattern.

Patterns are passed unchanged to rsync or rclone, so use the selected backend's
pattern syntax. These options work in both transfer directions. They do not
change the default dry-run behavior.

Examples
--------
Preview Windows -> WSL synchronization::

    python3 data/sync_data.py pull \
        --windows-dir /mnt/d/Local/.../FlightTest

Preview Windows -> WSL with the predefined rsync exclusions::

    python3 data/sync_data.py pull --windows-dir /mnt/d/Local/.../FlightTest --default-excludes

Execute Windows -> WSL synchronization::

    python3 data/sync_data.py pull \
        --windows-dir /mnt/d/Local/.../FlightTest \
        --execute

Preview OneDrive -> WSL download::

    python3 data/sync_data.py download \
        --onedrive-dir motif_onedrive:MOTIF/FlightTest

Preview OneDrive -> WSL with predefined and custom rclone exclusions::

    python3 data/sync_data.py download --onedrive-dir motif_onedrive:MOTIF/FlightTest --default-excludes --exclude "**/CalibOpt_pipeline/**"

Execute WSL -> OneDrive upload::

    python3 data/sync_data.py upload \
        --onedrive-dir motif_onedrive:MOTIF/FlightTest \
        --execute
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_WINDOWS_DIR = Path("/mnt/d/Local/FlightTest")

# Relative to the current working directory.
DEFAULT_WSL_DIR = Path("data/flight_test")


# ---------------------------------------------------------------------------
# Exclude rules
# ---------------------------------------------------------------------------

# rsync patterns
DEFAULT_RSYNC_EXCLUDES = [
    "archives/"
]

# rclone patterns
DEFAULT_RCLONE_EXCLUDES = [
    "/260827_flight_logs_purt_test/**",
    "**/videos/**",
    "**/SAM3_pipeline/**",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_local_path(path: Path) -> Path:
    """Expand ~ and convert a local path to an absolute path."""
    return path.expanduser().resolve()


def check_command(command: str) -> None:
    """Ensure that an external command exists."""
    if shutil.which(command) is None:
        if command == "rsync":
            raise RuntimeError(
                "rsync is not installed.\n"
                "Install it with:\n"
                "  sudo apt update && sudo apt install rsync"
            )

        if command == "rclone":
            raise RuntimeError(
                "rclone is not installed.\n"
                "Install it with:\n"
                "  sudo apt update && sudo apt install rclone\n\n"
                "Then configure OneDrive with:\n"
                "  rclone config"
            )

        raise RuntimeError(f"Required command not found: {command}")


def print_command(cmd: list[str]) -> None:
    """Print command in a shell-readable form."""
    print("\nCommand:")
    print(" ".join(f'"{arg}"' if " " in arg else arg for arg in cmd))
    print()


# ---------------------------------------------------------------------------
# rsync: Windows <-> WSL
# ---------------------------------------------------------------------------

def build_rsync_command(
    source: Path,
    destination: Path,
    *,
    dry_run: bool,
    delete: bool,
    excludes: list[str] | None = None,
) -> list[str]:
    """Build an rsync command with only the caller-selected exclusions."""

    cmd = [
        "rsync",
        "-avh",
        "--no-perms",
        "--no-owner",
        "--no-group",
        "--info=progress2",
        "--itemize-changes",
    ]

    if dry_run:
        cmd.append("--dry-run")

    if delete:
        cmd.append("--delete")

    for pattern in excludes or []:
        cmd.extend(["--exclude", pattern])

    # Trailing "/" means copy the CONTENTS of the source directory.
    cmd.extend(
        [
            f"{source}/",
            f"{destination}/",
        ]
    )

    return cmd


def sync_windows_wsl(
    direction: str,
    windows_dir: Path,
    wsl_dir: Path,
    *,
    dry_run: bool,
    delete: bool,
    excludes: list[str] | None = None,
) -> None:
    """Run pull or push with the supplied rsync exclusions."""

    check_command("rsync")

    windows_dir = resolve_local_path(windows_dir)
    wsl_dir = resolve_local_path(wsl_dir)

    if direction == "pull":
        source = windows_dir
        destination = wsl_dir
        description = "Windows D: -> WSL"

    elif direction == "push":
        source = wsl_dir
        destination = windows_dir
        description = "WSL -> Windows D:"

    else:
        raise ValueError(f"Unknown direction: {direction}")

    if not source.exists():
        raise FileNotFoundError(
            f"Source directory does not exist:\n{source}"
        )

    destination.mkdir(parents=True, exist_ok=True)

    cmd = build_rsync_command(
        source,
        destination,
        dry_run=dry_run,
        delete=delete,
        excludes=excludes,
    )

    print("=" * 72)
    print(f"Operation      : {direction}")
    print(f"Direction      : {description}")
    print(f"Source         : {source}")
    print(f"Destination    : {destination}")
    print(f"Dry run        : {dry_run}")
    print(f"Delete         : {delete}")
    print("=" * 72)

    if excludes:
        print("\nExcluded:")
        for pattern in excludes:
            print(f"  - {pattern}")
    else:
        print("\nExcluded: none")

    print_command(cmd)

    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# rclone: OneDrive <-> WSL
# ---------------------------------------------------------------------------

def build_rclone_command(
    source: str,
    destination: str,
    *,
    dry_run: bool,
    delete: bool,
    excludes: list[str] | None = None,
) -> list[str]:
    """Build an rclone copy or sync command with selected exclusions."""

    # copy:
    #   Copy new/changed files, but do not delete destination-only files.
    #
    # sync:
    #   Make destination match source and delete destination-only files.
    operation = "sync" if delete else "copy"

    cmd = [
        "rclone",
        operation,
        source,
        destination,
        "--progress",
        "--stats-one-line",
    ]

    if dry_run:
        cmd.append("--dry-run")

    for pattern in excludes or []:
        cmd.extend(["--exclude", pattern])

    return cmd


def validate_onedrive_path(path: str) -> None:
    """Reject local paths before rclone can silently copy onto the local disk."""
    remote, separator, remote_path = path.partition(":")
    if (
        not separator
        or not remote
        or remote != remote.strip()
        or remote in (".", "..")
        or any(char in remote for char in ("/", "\\"))
        or remote_path.startswith("//")
        or (len(remote) == 1 and remote.isalpha() and remote_path.startswith(("/", "\\")))
    ):
        raise ValueError(
            "--onedrive-dir must use a named rclone remote: REMOTE:path/to/folder.\n"
            "A path without ':' is local and would not transfer data to OneDrive.\n"
            "List configured remotes with: rclone listremotes\n"
            "Example: FIRE_moonshot_classifier:DARPA_FIRE/FIRE_Moonshot/FlightTest"
        )


def sync_onedrive_wsl(
    direction: str,
    onedrive_dir: str,
    wsl_dir: Path,
    *,
    dry_run: bool,
    delete: bool,
    excludes: list[str] | None = None,
) -> None:
    """Run download or upload with the supplied rclone exclusions."""

    validate_onedrive_path(onedrive_dir)
    check_command("rclone")

    wsl_dir = resolve_local_path(wsl_dir)

    if direction == "download":
        source = onedrive_dir
        destination = str(wsl_dir)
        description = "OneDrive -> WSL"

        wsl_dir.mkdir(parents=True, exist_ok=True)

    elif direction == "upload":
        if not wsl_dir.exists():
            raise FileNotFoundError(
                f"WSL source directory does not exist:\n{wsl_dir}"
            )

        source = str(wsl_dir)
        destination = onedrive_dir
        description = "WSL -> OneDrive"

    else:
        raise ValueError(f"Unknown direction: {direction}")

    cmd = build_rclone_command(
        source,
        destination,
        dry_run=dry_run,
        delete=delete,
        excludes=excludes,
    )

    print("=" * 72)
    print(f"Operation      : {direction}")
    print(f"Direction      : {description}")
    print(f"Source         : {source}")
    print(f"Destination    : {destination}")
    print(f"Dry run        : {dry_run}")
    print(f"Delete         : {delete}")
    print("=" * 72)

    if excludes:
        print("\nExcluded:")
        for pattern in excludes:
            print(f"  - {pattern}")
    else:
        print("\nExcluded: none")

    print_command(cmd)

    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add transfer flags, including optional default and custom exclusions."""

    parser.add_argument(
        "--windows-dir",
        type=Path,
        default=DEFAULT_WINDOWS_DIR,
        help=(
            "Windows FlightTest directory. "
            "Directory mounted in WSL, e.g. /mnt/d/.../FlightTest"
        )
    )

    parser.add_argument(
        "--wsl-dir",
        type=Path,
        default=DEFAULT_WSL_DIR,
        help=(
            "WSL FlightTest directory. "
            f"Default: {DEFAULT_WSL_DIR} "
            "(relative to the current working directory)."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually transfer files. Default behavior is dry-run.",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help=(
            "Mirror source to destination by deleting destination-only files. "
            "Use with caution."
        ),
    )

    parser.add_argument(
        "--default-excludes",
        action="store_true",
        help="Apply the predefined exclude patterns for the selected backend.",
    )

    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Additional exclude pattern; may be specified multiple times.",
    )


def main() -> None:
    """Select rsync or rclone defaults only when explicitly requested."""

    parser = argparse.ArgumentParser(
        description=(
            "Synchronize MOTIF FlightTest data between "
            "Windows, WSL, and OneDrive."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # -----------------------------------------------------------------------
    # pull
    # -----------------------------------------------------------------------

    pull_parser = subparsers.add_parser(
        "pull",
        help="Copy FlightTest data from Windows to WSL.",
    )

    add_common_arguments(pull_parser)

    # -----------------------------------------------------------------------
    # push
    # -----------------------------------------------------------------------

    push_parser = subparsers.add_parser(
        "push",
        help="Copy FlightTest data from WSL to Windows.",
    )

    add_common_arguments(push_parser)

    # -----------------------------------------------------------------------
    # download
    # -----------------------------------------------------------------------

    download_parser = subparsers.add_parser(
        "download",
        help="Download FlightTest data from OneDrive to WSL.",
    )

    download_parser.add_argument(
        "--onedrive-dir",
        required=True,
        help=(
            "rclone OneDrive path, e.g. "
            "motif_onedrive:MOTIF/FlightTest"
        ),
    )

    add_common_arguments(download_parser)

    # -----------------------------------------------------------------------
    # upload
    # -----------------------------------------------------------------------

    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload FlightTest data from WSL to OneDrive.",
    )

    upload_parser.add_argument(
        "--onedrive-dir",
        required=True,
        help=(
            "rclone OneDrive path, e.g. "
            "motif_onedrive:MOTIF/FlightTest"
        ),
    )

    add_common_arguments(upload_parser)

    args = parser.parse_args()

    dry_run = not args.execute

    if args.command in {"pull", "push"}:
        excludes = [*DEFAULT_RSYNC_EXCLUDES] if args.default_excludes else []
        excludes.extend(args.exclude)
        sync_windows_wsl(
            direction=args.command,
            windows_dir=args.windows_dir,
            wsl_dir=args.wsl_dir,
            dry_run=dry_run,
            delete=args.delete,
            excludes=excludes,
        )

    elif args.command in {"download", "upload"}:
        excludes = [*DEFAULT_RCLONE_EXCLUDES] if args.default_excludes else []
        excludes.extend(args.exclude)
        sync_onedrive_wsl(
            direction=args.command,
            onedrive_dir=args.onedrive_dir,
            wsl_dir=args.wsl_dir,
            dry_run=dry_run,
            delete=args.delete,
            excludes=excludes,
        )


if __name__ == "__main__":
    main()