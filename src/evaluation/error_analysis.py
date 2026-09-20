from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from src.evaluation.metrics import answer_matches


def identify_errors(prediction_records: list[dict]) -> list[dict]:
    errors = []
    for record in prediction_records:
        if not answer_matches(record.get("predicted_answer"), record.get("expected_answer"), record.get("choices")):
            enriched = dict(record)
            enriched.setdefault("analysis_category", "Other")
            errors.append(enriched)
    return errors


def summarize_errors(errors: list[dict]) -> dict:
    by_type = Counter(error.get("question_type") or "unknown" for error in errors)
    by_category = Counter(error.get("analysis_category") or "Other" for error in errors)
    return {"total_errors": len(errors), "by_question_type": dict(by_type), "by_analysis_category": dict(by_category)}


def write_jsonl(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
