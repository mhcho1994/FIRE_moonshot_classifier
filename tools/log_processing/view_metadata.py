#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_metadata(
    flight_log_dir: Path,
) -> dict:
    """Load metadata.yaml from a flight log directory.

    Expected structure:

        <flight_log_dir>/
        └── source/
            └── metadata.yaml
    """

    flight_log_dir = flight_log_dir.expanduser().resolve()
    if not flight_log_dir.is_dir():
        raise FileNotFoundError(
            f"Flight log directory was not found:\n  {flight_log_dir}"
        )

    metadata_path = (
        flight_log_dir
        / "source"
        / "metadata.yaml"
    ).expanduser().resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(
            "metadata.yaml was not found:\n"
            f"  {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    if metadata is None:
        raise ValueError(
            f"metadata.yaml is empty:\n  {metadata_path}"
        )

    return metadata


def print_metadata(
    flight_log_dir: Path,
) -> None:
    """Load and print metadata for the selected flight log directory."""

    metadata_path = (
        flight_log_dir
        / "source"
        / "metadata.yaml"
    ).expanduser().resolve()

    metadata = load_metadata(flight_log_dir)

    print("=" * 72)
    print(f"Flight log dir: {flight_log_dir.expanduser().resolve()}")
    print(f"Metadata path : {metadata_path}")
    print("=" * 72)
    print()

    print(
        yaml.safe_dump(
            metadata,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read source/metadata.yaml from a flight log directory."
        )
    )

    parser.add_argument(
        "--flight-log-dir",
        type=Path,
        required=True,
        help=(
            "Path to the flight log directory, e.g. "
            "data/flight_test/260501_flight_logs"
        ),
    )

    args = parser.parse_args()

    try:
        print_metadata(args.flight_log_dir)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
