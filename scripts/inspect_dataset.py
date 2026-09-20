from __future__ import annotations

import _path_setup  # noqa: F401

from collections import Counter

from src.data.dataset_loader import load_all_splits
from src.utils.audio import audio_info


def main() -> None:
    splits = load_all_splits()
    for split, records in splits.items():
        types = Counter(record.question_type or "unknown" for record in records)
        print(f"{split}: {len(records)} QA records, {len({r.audio_path for r in records})} audio files")
        print(dict(sorted(types.items())))
    sample = next(iter(splits["train"]))
    print("Sample audio info:", audio_info(sample.audio_path))
    print("Sample question:", sample.question)


if __name__ == "__main__":
    main()
