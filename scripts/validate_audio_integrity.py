"""Validate every audio file referenced by the prepared manifests using the
actual runtime audio stack (soundfile/librosa), not Python's stdlib `wave`.

A prior stdlib-`wave`-based check misreported ~1,292 files as "corrupted";
most were actually decodable (either valid IEEE-float WAV, which `wave`
cannot parse at all, or MPEG/MP3 data saved with a .wav extension, which
libsndfile >= 1.2.0 decodes natively). This script re-checks with the real
loader the training/inference pipeline uses and reports which files, if any,
genuinely fail to decode.
"""

from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
from collections import Counter

from src.data.audio_integrity import check_audio_file
from src.data.dataset_loader import load_all_splits
from src.utils.config import load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manifest-referenced audio with the real audio stack.")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--output", default="results/metrics/audio_integrity.json")
    parser.add_argument("--no-decode", action="store_true", help="Skip the full librosa decode pass (header check only, faster).")
    args = parser.parse_args()

    dataset_cfg = load_yaml(args.dataset_config)
    splits = load_all_splits(dataset_cfg["dataset_root"], schema=dataset_cfg.get("schema"))

    seen: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            seen.setdefault(record.audio_path, split)

    results = []
    for path, split in seen.items():
        result = check_audio_file(path, decode_check=not args.no_decode)
        entry = result.to_dict()
        entry["split"] = split
        results.append(entry)

    by_container = Counter(r["container"] for r in results)
    undecodable = [r for r in results if not r["decodable"]]
    extension_mismatches = [r for r in results if r["extension_mismatch"]]
    extension_mismatches_by_split = Counter(r["split"] for r in extension_mismatches)

    summary = {
        "total_unique_audio_checked": len(results),
        "decodable": len(results) - len(undecodable),
        "undecodable": len(undecodable),
        "undecodable_examples": undecodable[:20],
        "container_type_counts": dict(by_container),
        "extension_mismatch_total": len(extension_mismatches),
        "extension_mismatch_by_split": dict(extension_mismatches_by_split),
        "extension_mismatch_examples": extension_mismatches[:10],
        "decode_check_performed": not args.no_decode,
        "note": (
            "extension_mismatch means the file's magic bytes are MPEG/MP3 "
            "but the filename ends in .wav. These are still decodable by "
            "the pinned soundfile/libsndfile version used in this project "
            "(see requirements.txt) and are NOT transcoded -- see "
            "docs/dataset.md for the verification and the required "
            "soundfile/libsndfile version floor."
        ),
    }

    save_json({"summary": summary, "files": results}, args.output)
    print_summary = {k: v for k, v in summary.items() if k not in ("undecodable_examples", "extension_mismatch_examples")}
    import json

    print(json.dumps(print_summary, indent=2))
    if undecodable:
        raise SystemExit(f"{len(undecodable)} manifest-referenced audio files failed to decode. See {args.output}.")


if __name__ == "__main__":
    main()
