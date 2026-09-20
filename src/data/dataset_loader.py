from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from torch.utils.data import Dataset
except ImportError:  # torch is only required to actually train/collate, not
    # for manifest loading, dataset validation, or leakage checks -- keep
    # those runnable in a lightweight environment without the full ML stack.
    class Dataset:  # type: ignore[no-redef]
        pass

from src.utils.paths import resolve_project_path


# Default manifest field names. configs/dataset.yaml's `schema:` block can
# override any of these (e.g. if a future dataset's prepared manifest uses
# different key names) -- see AudioQARecord.from_dict / load_manifest.
DEFAULT_SCHEMA = {
    "question_field": "question",
    "choices_field": "choices",
    "answer_field": "answer",
    "audio_id_field": "audio_id",
    "audio_path_field": "audio_path",
    "question_type_field": "question_type",
}
# record_id/split are structural (assigned by the manifest builder, not part
# of the raw QA schema) and are always required under these exact names.
ALWAYS_REQUIRED_FIELDS = {"record_id", "split"}


@dataclass(frozen=True)
class AudioQARecord:
    record_id: str
    audio_id: str
    audio_path: str
    question: str
    answer: str
    split: str
    choices: list[str] | None = None
    question_type: str | None = None
    source_split: str | None = None
    metadata_file: str | None = None
    original_audio_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict, schema: dict | None = None) -> "AudioQARecord":
        schema = {**DEFAULT_SCHEMA, **(schema or {})}
        required = ALWAYS_REQUIRED_FIELDS | {
            schema["audio_id_field"],
            schema["audio_path_field"],
            schema["question_field"],
            schema["answer_field"],
        }
        missing = required.difference(data)
        if missing:
            raise ValueError(f"Manifest record is missing required fields: {sorted(missing)}")
        choices_value = data.get(schema["choices_field"])
        return cls(
            record_id=str(data["record_id"]),
            audio_id=str(data[schema["audio_id_field"]]),
            audio_path=str(data[schema["audio_path_field"]]),
            question=str(data[schema["question_field"]]),
            answer=str(data[schema["answer_field"]]),
            split=str(data["split"]),
            choices=list(choices_value) if choices_value is not None else None,
            question_type=data.get(schema["question_type_field"]),
            source_split=data.get("source_split"),
            metadata_file=data.get("metadata_file"),
            original_audio_url=data.get("original_audio_url"),
        )

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with resolve_project_path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def load_manifest(path: str | Path, validate_audio: bool = True, schema: dict | None = None) -> list[AudioQARecord]:
    records = [AudioQARecord.from_dict(item, schema=schema) for item in read_jsonl(path)]
    if validate_audio:
        missing = [record.audio_path for record in records if not resolve_project_path(record.audio_path).exists()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"{len(missing)} manifest audio paths are missing. First missing: {preview}")
    return records


def load_split(split: str, dataset_root: str | Path = "dataset/dcase_part1", schema: dict | None = None) -> list[AudioQARecord]:
    split_path = Path(dataset_root) / "splits" / f"{split}.jsonl"
    return load_manifest(split_path, schema=schema)


def load_all_splits(dataset_root: str | Path = "dataset/dcase_part1", schema: dict | None = None) -> dict[str, list[AudioQARecord]]:
    return {split: load_split(split, dataset_root, schema=schema) for split in ("train", "validation", "test")}


class AudioQADataset(Dataset):
    def __init__(self, records: Iterable[AudioQARecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index].to_dict()

