"""Verifies the P0-3 fix: leakage detection must key on audio_id, not local
file path (paths never collide across train/dev source directories even
when the underlying recording is duplicated), and remediation must remove
the right records while leaving the reference split untouched.
"""

from __future__ import annotations

import unittest

from src.data.splitting import assign_audio_level_validation, check_audio_leakage, remove_overlapping_audio


class LeakageDetectionTests(unittest.TestCase):
    def test_path_based_comparison_would_miss_this_but_audio_id_catches_it(self) -> None:
        # Same underlying recording (audio_id "us1RbA4r9t"), copied under two
        # different directories -- exactly the real-world pattern found in
        # this project's dataset (train/ and dev/ are separate directories).
        train = [{"audio_id": "us1RbA4r9t", "audio_path": "audio/train/us1RbA4r9t.wav", "question": "Q1"}]
        test = [{"audio_id": "us1RbA4r9t", "audio_path": "audio/dev/us1RbA4r9t.wav", "question": "Q2"}]

        # Sanity: paths genuinely differ.
        self.assertNotEqual(train[0]["audio_path"], test[0]["audio_path"])

        leakage = check_audio_leakage({"train": train, "test": test})
        self.assertIn("test__train", leakage)
        self.assertEqual(leakage["test__train"], ["us1RbA4r9t"])

    def test_no_leakage_when_audio_ids_are_disjoint(self) -> None:
        train = [{"audio_id": "a1", "audio_path": "audio/train/a1.wav", "question": "Q1"}]
        test = [{"audio_id": "b1", "audio_path": "audio/dev/b1.wav", "question": "Q2"}]
        self.assertEqual(check_audio_leakage({"train": train, "test": test}), {})


class RemediationTests(unittest.TestCase):
    def test_remove_overlapping_audio_keeps_reference_untouched(self) -> None:
        test_records = [{"audio_id": "shared", "audio_path": "audio/dev/shared.wav"}]
        train_records = [
            {"audio_id": "shared", "audio_path": "audio/train/shared.wav"},
            {"audio_id": "unique", "audio_path": "audio/train/unique.wav"},
        ]
        cleaned, removed = remove_overlapping_audio(test_records, train_records)
        self.assertEqual([r["audio_id"] for r in cleaned], ["unique"])
        self.assertEqual([r["audio_id"] for r in removed], ["shared"])
        # reference (test) split must never be mutated by remediation
        self.assertEqual(test_records, [{"audio_id": "shared", "audio_path": "audio/dev/shared.wav"}])

    def test_remediation_result_has_zero_residual_leakage(self) -> None:
        test_records = [{"audio_id": "shared", "audio_path": "audio/dev/shared.wav", "question": "Q"}]
        train_records = [
            {"audio_id": "shared", "audio_path": "audio/train/shared.wav", "question": "Q2"},
            {"audio_id": "unique", "audio_path": "audio/train/unique.wav", "question": "Q3"},
        ]
        cleaned_train, _ = remove_overlapping_audio(test_records, train_records)
        leakage = check_audio_leakage({"train": cleaned_train, "test": test_records})
        self.assertEqual(leakage, {})


class AudioLevelSplitTests(unittest.TestCase):
    def test_assignment_is_deterministic(self) -> None:
        audio_ids = [f"id{i:03d}" for i in range(100)]
        first = assign_audio_level_validation(audio_ids, validation_every_n=10)
        second = assign_audio_level_validation(list(reversed(audio_ids)), validation_every_n=10)
        self.assertEqual(first, second)  # order-independent: sorts internally
        self.assertEqual(len(first), 10)

    def test_rejects_invalid_n(self) -> None:
        with self.assertRaises(ValueError):
            assign_audio_level_validation(["a", "b"], validation_every_n=1)


if __name__ == "__main__":
    unittest.main()
