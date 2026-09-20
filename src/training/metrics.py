from __future__ import annotations

from src.evaluation.metrics import exact_match_accuracy


def compute_eval_metrics(predictions: list[str], references: list[str]) -> dict:
    return exact_match_accuracy(predictions, references)

