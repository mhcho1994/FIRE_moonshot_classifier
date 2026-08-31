"""Backward-compatible wrapper for ``fireclassify feature-build``."""
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_moonshot_classifier.cli import main as cli_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["feature-build", *args])


if __name__ == "__main__":
    raise SystemExit(main())
