from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import json

from src.evaluation.evaluate import evaluate_prediction_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved prediction records.")
    parser.add_argument("--predictions", default="results/predictions/predictions.jsonl")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    records = []
    with open(args.predictions, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    print(json.dumps(evaluate_prediction_records(records, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
