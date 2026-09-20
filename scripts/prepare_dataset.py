from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import json
from pathlib import Path

from src.data.validation import validate_splits
from src.utils.config import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prepared DCASE Part 1 manifests.")
    parser.add_argument("--dataset-root", default="dataset/dcase_part1")
    parser.add_argument("--report-path", default="results/metrics/data_split_validation.json")
    args = parser.parse_args()
    report = validate_splits(args.dataset_root)
    Path(args.dataset_root, "splits", "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_json(report, args.report_path)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit("Dataset split validation failed.")


if __name__ == "__main__":
    main()
