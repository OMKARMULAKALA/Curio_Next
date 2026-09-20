from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import json

from src.data.validation import validate_splits
from src.utils.config import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate audio-level split integrity.")
    parser.add_argument("--dataset-root", default="dataset/dcase_part1")
    parser.add_argument("--output", default="results/metrics/data_split_validation.json")
    args = parser.parse_args()
    report = validate_splits(args.dataset_root)
    save_json(report, args.output)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit("Split validation failed.")


if __name__ == "__main__":
    main()
