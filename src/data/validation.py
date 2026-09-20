from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.data.dataset_loader import AudioQARecord, load_all_splits
from src.data.splitting import check_audio_leakage, detailed_leakage_report
from src.utils.paths import resolve_project_path


def question_type_distribution(records: list[AudioQARecord]) -> dict[str, dict]:
    total = len(records)
    counts = Counter(record.question_type or "unknown" for record in records)
    return {
        key: {"count": count, "percentage": round((100.0 * count / total) if total else 0.0, 4)}
        for key, count in sorted(counts.items())
    }


def _strip_trailing_punctuation(text: str) -> str:
    return text.strip().rstrip(",;")


def answer_choice_consistency(records: list[AudioQARecord]) -> dict:
    """Check whether ``answer`` appears verbatim among ``choices``.

    A subset of the raw DCASE metadata has a trailing-comma inconsistency:
    the `choice` list entries end with a stray "," that the `answer` field
    does not have (e.g. choice "C. text," vs answer "C. text"). That is a
    pure formatting artifact and is resolved by stripping trailing
    punctuation before comparing. Records that still don't match after that
    normalization are genuine source-metadata defects (verified by manual
    inspection -- see docs/dataset.md) and are reported separately rather
    than silently dropped.
    """
    raw_mismatch = []
    normalized_mismatch = []
    for record in records:
        if not record.choices:
            continue
        if record.answer in record.choices:
            continue
        raw_mismatch.append(record.record_id)
        normalized_choices = {_strip_trailing_punctuation(c) for c in record.choices}
        if _strip_trailing_punctuation(record.answer) not in normalized_choices:
            normalized_mismatch.append(record.record_id)
    return {
        "raw_mismatch_count": len(raw_mismatch),
        "raw_mismatch_examples": raw_mismatch[:20],
        "resolved_by_trailing_punctuation_normalization": len(raw_mismatch) - len(normalized_mismatch),
        "genuinely_malformed_count": len(normalized_mismatch),
        "genuinely_malformed_examples": normalized_mismatch[:20],
    }


def duplicate_audio_summary(records: list[AudioQARecord]) -> dict:
    counts = Counter(record.audio_id for record in records)
    duplicates = {audio_id: count for audio_id, count in sorted(counts.items()) if count > 1}
    return {
        "duplicate_audio_ids": len(duplicates),
        "duplicate_qa_records": sum(count - 1 for count in duplicates.values()),
        "examples": dict(list(duplicates.items())[:20]),
    }


def validate_splits(dataset_root: str | Path = "dataset/dcase_part1", compute_content_hash: bool = True) -> dict:
    """Comprehensive, audio-id-level split validation.

    Leakage is checked by ``audio_id`` (the underlying-recording identifier),
    not local file path -- train/dev source audio lives under different
    directories, so path comparison alone cannot detect the same recording
    appearing in more than one split. When ``compute_content_hash`` is true
    and any overlap is found, the actual file bytes are hashed to confirm it
    is genuinely the same recording rather than a coincidental id collision,
    and any exact (audio, question) duplicate is flagged separately.
    """
    splits = load_all_splits(dataset_root)
    report: dict = {"splits": {}, "overlaps": {}, "leakage_detail": [], "question_types_missing_from_train": []}
    audio_by_split = {split: {record.audio_id for record in records} for split, records in splits.items()}
    train_types = {record.question_type for record in splits["train"]}

    for split, records in splits.items():
        missing_audio = [record.audio_path for record in records if not resolve_project_path(record.audio_path).exists()]
        report["splits"][split] = {
            "records": len(records),
            "unique_audio_count": len(audio_by_split[split]),
            "missing_audio_references": len(missing_audio),
            "missing_audio_examples": missing_audio[:20],
            "question_type_distribution": question_type_distribution(records),
            "duplicates": duplicate_audio_summary(records),
            "answer_choice_consistency": answer_choice_consistency(records),
        }

    split_record_dicts = {split: [record.to_dict() for record in records] for split, records in splits.items()}
    leakage = check_audio_leakage(split_record_dicts)
    for pair, overlap_ids in leakage.items():
        report["overlaps"][pair] = {"count": len(overlap_ids), "audio_ids": overlap_ids[:50]}
    names = sorted(splits)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            report["overlaps"].setdefault(f"{left}__{right}", {"count": 0, "audio_ids": []})

    if leakage:
        detail_entries = detailed_leakage_report(split_record_dicts, compute_content_hash=compute_content_hash)
        report["leakage_detail"] = [entry.to_dict() for entry in detail_entries]
        report["leakage_summary"] = {
            "total_overlapping_audio_ids": sum(v["count"] for v in report["overlaps"].values()),
            "content_verified_identical": sum(1 for e in detail_entries if e.content_identical),
            "exact_qa_duplicates": sum(1 for e in detail_entries if e.exact_qa_duplicate),
        }

    for split in ("validation", "test"):
        missing_types = sorted({record.question_type for record in splits[split]} - train_types)
        report["question_types_missing_from_train"].append({"split": split, "question_types": missing_types})

    report["ok"] = (
        all(item["missing_audio_references"] == 0 for item in report["splits"].values())
        and all(item["count"] == 0 for item in report["overlaps"].values())
    )
    return report
