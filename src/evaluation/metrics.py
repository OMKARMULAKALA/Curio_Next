from __future__ import annotations

from collections import defaultdict
import re


# Requires punctuation immediately after the letter (e.g. "C.", "C)", "C:")
# rather than a bare standalone letter -- a bare match would false-positive
# on the English article "a" (or "i" is safe, but "a"/"b" are common short
# words) appearing anywhere in a free-form model response.
OPTION_RE = re.compile(r"\b([A-D])[\).:-]", flags=re.IGNORECASE)


def normalize_answer(text: str | None) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"^(the\s+)?answer\s+is\s+", "", text)
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return " ".join(text.split())


def extract_option_label(text: str | None) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0].upper() in "ABCD" and stripped[1] in ".):":
        return stripped[0].upper()
    match = OPTION_RE.search(stripped)
    if match:
        return match.group(1).upper()
    return None


def answer_matches(prediction: str | None, reference: str | None, choices: list[str] | None = None) -> bool:
    pred_label = extract_option_label(prediction)
    ref_label = extract_option_label(reference)
    if pred_label and ref_label:
        return pred_label == ref_label
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    if normalized_prediction == normalized_reference:
        return True
    if choices and pred_label:
        for choice in choices:
            if extract_option_label(choice) == pred_label and normalize_answer(choice) == normalized_reference:
                return True
    return False


def exact_match_accuracy(predictions: list[str], references: list[str], choices: list[list[str] | None] | None = None) -> dict:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if choices is None:
        choices = [None] * len(predictions)
    correct = sum(answer_matches(p, r, c) for p, r, c in zip(predictions, references, choices))
    total = len(references)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct_count": correct,
        "incorrect_count": total - correct,
        "total_count": total,
    }


def accuracy_by_field(records: list[dict], field: str = "question_type") -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record.get(field) or "unknown")].append(record)
    return {
        key: exact_match_accuracy(
            [item["predicted_answer"] for item in items],
            [item["expected_answer"] for item in items],
            [item.get("choices") for item in items],
        )
        for key, items in sorted(groups.items())
    }
