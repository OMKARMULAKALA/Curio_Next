from __future__ import annotations

import unittest

from src.data.dataset_loader import AudioQARecord
from src.utils.paths import project_root, resolve_project_path


class AudioQARecordSchemaTests(unittest.TestCase):
    def test_default_schema_reads_standard_field_names(self) -> None:
        data = {
            "record_id": "r1",
            "audio_id": "a1",
            "audio_path": "dataset/dcase_part1/audio/train/a1.wav",
            "question": "Q?",
            "answer": "A",
            "split": "train",
            "choices": ["A", "B"],
        }
        record = AudioQARecord.from_dict(data)
        self.assertEqual(record.audio_id, "a1")
        self.assertEqual(record.choices, ["A", "B"])

    def test_missing_required_field_raises(self) -> None:
        data = {"record_id": "r1", "audio_id": "a1", "question": "Q?", "answer": "A", "split": "train"}
        with self.assertRaises(ValueError):
            AudioQARecord.from_dict(data)  # missing audio_path

    def test_configs_dataset_yaml_schema_is_actually_used(self) -> None:
        # P1 fix: configs/dataset.yaml's `schema:` block must be a real,
        # honored field-name mapping, not decorative. Simulate a manifest
        # produced under a *different* schema and confirm from_dict()
        # resolves it correctly when given that mapping.
        custom_schema = {
            "question_field": "q_text",
            "choices_field": "options",
            "answer_field": "correct_option",
            "audio_id_field": "clip_id",
            "audio_path_field": "clip_path",
            "question_type_field": "category",
        }
        data = {
            "record_id": "r1",
            "split": "train",
            "clip_id": "a1",
            "clip_path": "dataset/dcase_part1/audio/train/a1.wav",
            "q_text": "Q?",
            "correct_option": "A",
            "options": ["A", "B"],
            "category": "counting",
        }
        record = AudioQARecord.from_dict(data, schema=custom_schema)
        self.assertEqual(record.audio_id, "a1")
        self.assertEqual(record.audio_path, "dataset/dcase_part1/audio/train/a1.wav")
        self.assertEqual(record.question, "Q?")
        self.assertEqual(record.answer, "A")
        self.assertEqual(record.question_type, "counting")


class PathResolutionTests(unittest.TestCase):
    def test_relative_path_resolves_against_project_root(self) -> None:
        resolved = resolve_project_path("dataset/dcase_part1/splits/train.jsonl")
        self.assertTrue(str(resolved).startswith(str(project_root())))
        self.assertTrue(resolved.is_absolute())

    def test_absolute_path_is_returned_unchanged(self) -> None:
        absolute = project_root() / "README.md"
        self.assertEqual(resolve_project_path(str(absolute)), absolute)


if __name__ == "__main__":
    unittest.main()
