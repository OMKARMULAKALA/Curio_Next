"""Reproducible pipeline: raw DCASE metadata -> prepared, leakage-free manifests.

This is the authoritative, runnable replacement for the manifest files that
previously existed only as a manually-produced (and irreproducible) snapshot
under dataset/dcase_part1/splits/. Running this script end to end from the
raw per-record metadata JSON files always regenerates the same
train.jsonl / validation.jsonl / test.jsonl given the same raw inputs.

Pipeline:
  1. Discover raw metadata JSON under dataset/dcase_part1/metadata/{train,dev}.
  2. Normalize each record (src.data.preprocessing.normalize_raw_record).
  3. Keep only records whose referenced audio file actually exists locally.
  4. dev records become the `test` split verbatim (source_split == "dev" is
     kept intact as the official held-out evaluation set).
  5. train records are split at the AUDIO level: every 10th unique audio_id
     (sorted) goes to `validation`, the rest to `train` -- this is the
     "official_train_audio_level_90_10_validation_official_dev_as_test"
     policy already named in configs/dataset.yaml.
  6. Leakage remediation: any audio_id in the (train + validation) pool that
     also appears in `test` is REMOVED from train/validation. `test` is never
     modified, so the official dev partition stays intact as the evaluation
     set.
  7. Validate: zero missing audio, zero duplicate record_ids, zero remaining
     cross-split audio_id overlap.
  8. Write dataset/dcase_part1/splits/{train,validation,test}.jsonl, after
     backing up whatever was there before.

Usage:
    python scripts/build_manifests.py
    python scripts/build_manifests.py --dry-run   # report only, write nothing
"""

from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.data.preprocessing import normalize_raw_record, read_raw_json
from src.data.splitting import assign_audio_level_validation, check_audio_leakage, remove_overlapping_audio
from src.utils.config import load_yaml, save_json
from src.utils.paths import resolve_project_path


def discover_and_normalize(dataset_root: Path, source_split: str, audio_root: Path) -> list[dict]:
    metadata_dir = dataset_root / "metadata" / source_split
    records: list[dict] = []
    for metadata_file in sorted(metadata_dir.glob("*.json")):
        raw = read_raw_json(metadata_file)
        rel_metadata_file = str(metadata_file.relative_to(resolve_project_path(".")))
        record = normalize_raw_record(raw, source_split, rel_metadata_file, audio_root)
        records.append(record)
    return records


def keep_resolvable(records: list[dict]) -> tuple[list[dict], list[dict]]:
    resolvable, missing = [], []
    for record in records:
        if resolve_project_path(record["audio_path"]).exists():
            resolvable.append(record)
        else:
            missing.append(record)
    return resolvable, missing


def assign_split_field(records: list[dict], split_name: str) -> list[dict]:
    out = []
    for record in records:
        r = dict(record)
        r["split"] = split_name
        out.append(r)
    return out


def backup_existing_manifests(splits_dir: Path) -> Path | None:
    existing = [splits_dir / f"{name}.jsonl" for name in ("train", "validation", "test")]
    if not any(p.exists() for p in existing):
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = splits_dir / f"_backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        if p.exists():
            shutil.copy2(p, backup_dir / p.name)
    return backup_dir


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build(dataset_root: Path, validation_every_n: int = 10, dry_run: bool = False) -> dict:
    audio_root = dataset_root / "audio"
    splits_dir = dataset_root / "splits"

    raw_train = discover_and_normalize(dataset_root, "train", audio_root)
    raw_dev = discover_and_normalize(dataset_root, "dev", audio_root)

    train_pool_all, train_missing = keep_resolvable(raw_train)
    test_all, test_missing = keep_resolvable(raw_dev)

    train_audio_ids = sorted({r["audio_id"] for r in train_pool_all})
    validation_ids = assign_audio_level_validation(train_audio_ids, validation_every_n=validation_every_n)

    original_train = assign_split_field([r for r in train_pool_all if r["audio_id"] not in validation_ids], "train")
    original_validation = assign_split_field([r for r in train_pool_all if r["audio_id"] in validation_ids], "validation")
    test_records = assign_split_field(test_all, "test")

    test_audio_ids = {r["audio_id"] for r in test_records}
    cleaned_train, removed_from_train = remove_overlapping_audio(test_records, original_train)
    cleaned_validation, removed_from_validation = remove_overlapping_audio(test_records, original_validation)

    final_pairs = {
        "train": cleaned_train,
        "validation": cleaned_validation,
        "test": test_records,
    }
    final_leakage = check_audio_leakage({k: v for k, v in final_pairs.items()})

    summary = {
        "raw_metadata_counts": {"train": len(raw_train), "dev": len(raw_dev)},
        "missing_audio_dropped": {"train": len(train_missing), "dev": len(test_missing)},
        "validation_every_n": validation_every_n,
        "original_train_count": len(original_train),
        "original_validation_count": len(original_validation),
        "contaminated_train_audio_count": len(removed_from_train),
        "contaminated_validation_audio_count": len(removed_from_validation),
        "cleaned_train_count": len(cleaned_train),
        "cleaned_validation_count": len(cleaned_validation),
        "test_count": len(test_records),
        "test_unique_audio_count": len(test_audio_ids),
        "final_overlaps": {pair: len(ids) for pair, ids in final_leakage.items()},
        "final_leakage_clean": not final_leakage,
        "removed_train_audio_ids": sorted({r["audio_id"] for r in removed_from_train}),
        "removed_validation_audio_ids": sorted({r["audio_id"] for r in removed_from_validation}),
    }

    if final_leakage:
        raise RuntimeError(f"Remediation failed, overlaps remain: {final_leakage}")

    if not dry_run:
        backup_dir = backup_existing_manifests(splits_dir)
        summary["backup_dir"] = str(backup_dir) if backup_dir else None
        write_jsonl(cleaned_train, splits_dir / "train.jsonl")
        write_jsonl(cleaned_validation, splits_dir / "validation.jsonl")
        write_jsonl(test_records, splits_dir / "test.jsonl")
        save_json(summary, splits_dir / "build_manifests_summary.json")
    else:
        summary["backup_dir"] = None
        summary["dry_run"] = True

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild leakage-free train/validation/test manifests from raw DCASE metadata.")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--validation-every-n", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Compute and print the summary but do not write any files.")
    args = parser.parse_args()

    dataset_cfg = load_yaml(args.dataset_config)
    dataset_root = resolve_project_path(dataset_cfg["dataset_root"])

    summary = build(dataset_root, validation_every_n=args.validation_every_n, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
