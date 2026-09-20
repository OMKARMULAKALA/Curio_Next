from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.paths import resolve_project_path


def assign_audio_level_validation(audio_keys: list[str], validation_every_n: int = 10) -> set[str]:
    """Deterministically assign every Nth unique audio key to validation.

    ``audio_keys`` must be a stable identifier for the underlying recording
    (``audio_id``), not a local file path -- paths differ across the
    train/dev source directories even when the underlying audio is the same
    file, which previously hid leakage from :func:`check_audio_leakage`.
    """
    if validation_every_n <= 1:
        raise ValueError("validation_every_n must be greater than 1")
    return {key for index, key in enumerate(sorted(set(audio_keys))) if index % validation_every_n == 0}


def file_sha1(path: str | Path, chunk_size: int = 1 << 20) -> str:
    resolved = resolve_project_path(path)
    digest = hashlib.sha1()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_audio_leakage(split_records: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Return audio_id overlaps between every pair of splits.

    IMPORTANT: this compares ``audio_id`` (the underlying-recording
    identifier), not ``audio_path``. Local file paths live under
    ``audio/train/`` vs ``audio/dev/`` and therefore never collide even when
    the same recording is duplicated in both source partitions -- comparing
    paths silently hides real leakage. See docs/dataset.md for the
    audio-content verification that motivated this.
    """
    by_split = {split: {record["audio_id"] for record in records} for split, records in split_records.items()}
    leaks: dict[str, list[str]] = defaultdict(list)
    splits = sorted(by_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = sorted(by_split[left].intersection(by_split[right]))
            if overlap:
                leaks[f"{left}__{right}"] = overlap
    return dict(leaks)


@dataclass
class LeakageEntry:
    audio_id: str
    split_a: str
    split_b: str
    path_a: str
    path_b: str
    content_hash_a: str | None = None
    content_hash_b: str | None = None
    content_identical: bool | None = None
    question_a: list[str] = field(default_factory=list)
    question_b: list[str] = field(default_factory=list)
    exact_qa_duplicate: bool = False

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def detailed_leakage_report(
    split_records: dict[str, list[dict]],
    compute_content_hash: bool = True,
) -> list[LeakageEntry]:
    """Audio-id-level leakage report with optional byte-content verification.

    For every audio_id present in more than one split, records which splits
    it appears in, whether the underlying bytes are identical (proving it is
    the same recording rather than a coincidental id collision), and whether
    any (question, audio) pair is an exact duplicate across the two splits.
    """
    by_audio_id: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for split, records in split_records.items():
        for record in records:
            by_audio_id[record["audio_id"]][split].append(record)

    entries: list[LeakageEntry] = []
    for audio_id, per_split in by_audio_id.items():
        splits_present = sorted(per_split)
        if len(splits_present) < 2:
            continue
        for i, split_a in enumerate(splits_present):
            for split_b in splits_present[i + 1 :]:
                records_a = per_split[split_a]
                records_b = per_split[split_b]
                path_a = records_a[0]["audio_path"]
                path_b = records_b[0]["audio_path"]
                hash_a = hash_b = None
                identical = None
                if compute_content_hash:
                    try:
                        hash_a = file_sha1(path_a)
                        hash_b = file_sha1(path_b)
                        identical = hash_a == hash_b
                    except FileNotFoundError:
                        identical = None
                questions_a = [r["question"] for r in records_a]
                questions_b = [r["question"] for r in records_b]
                exact_dup = bool(set(q.strip() for q in questions_a) & set(q.strip() for q in questions_b))
                entries.append(
                    LeakageEntry(
                        audio_id=audio_id,
                        split_a=split_a,
                        split_b=split_b,
                        path_a=path_a,
                        path_b=path_b,
                        content_hash_a=hash_a,
                        content_hash_b=hash_b,
                        content_identical=identical,
                        question_a=questions_a,
                        question_b=questions_b,
                        exact_qa_duplicate=exact_dup,
                    )
                )
    return entries


def remove_overlapping_audio(
    reference_records: list[dict],
    candidate_records: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Remove any record from ``candidate_records`` whose audio_id also
    appears in ``reference_records``.

    Used to keep the official DCASE dev partition intact as the test set
    while removing train/validation contamination: call with
    ``reference_records=test_records`` and ``candidate_records=train_records``
    (and again for validation).

    Returns (cleaned_records, removed_records).
    """
    reference_ids = {r["audio_id"] for r in reference_records}
    cleaned = [r for r in candidate_records if r["audio_id"] not in reference_ids]
    removed = [r for r in candidate_records if r["audio_id"] in reference_ids]
    return cleaned, removed
