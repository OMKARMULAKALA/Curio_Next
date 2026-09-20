from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import confusion_matrix

from src.evaluation.error_analysis import identify_errors, write_jsonl
from src.evaluation.metrics import accuracy_by_field, exact_match_accuracy


def evaluate_prediction_records(records: list[dict], output_dir: str | Path = "results") -> dict:
    output_dir = Path(output_dir)
    predictions = [record["predicted_answer"] for record in records]
    references = [record["expected_answer"] for record in records]
    overall = exact_match_accuracy(predictions, references, [record.get("choices") for record in records])
    by_type = accuracy_by_field(records, "question_type")
    by_source_split = accuracy_by_field(records, "source_split")
    errors = identify_errors(records)

    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "error_analysis").mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics" / "overall_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    (output_dir / "metrics" / "question_type_metrics.json").write_text(json.dumps(by_type, indent=2), encoding="utf-8")
    (output_dir / "metrics" / "source_split_metrics.json").write_text(json.dumps(by_source_split, indent=2), encoding="utf-8")
    write_jsonl(records, output_dir / "predictions" / "predictions.jsonl")
    write_jsonl(errors, output_dir / "error_analysis" / "errors.jsonl")
    return {"overall": overall, "question_type": by_type, "source_split": by_source_split, "errors": len(errors)}


def label_confusion(records: list[dict]) -> dict:
    labels = sorted({record["expected_answer"] for record in records} | {record["predicted_answer"] for record in records})
    matrix = confusion_matrix(
        [record["expected_answer"] for record in records],
        [record["predicted_answer"] for record in records],
        labels=labels,
    )
    return {"labels": labels, "matrix": matrix.tolist()}
