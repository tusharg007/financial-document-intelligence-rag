"""Prepare optional financial datasets into standardized JSONL files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_loaders import prepare_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-ok", action="store_true", help="Also write demo eval pairs from bundled sample_data.py.")
    args = parser.parse_args()
    results = prepare_all(demo_ok=args.demo_ok)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
