"""Verifies the Fix-8 evaluation method (raw + option-label matching) and the
prediction record schema scripts/run_inference.py produces is exactly what
src/evaluation/* expects. Includes the real edge cases found in the actual
test manifest (Fix 15): a trailing-comma formatting artifact and a
letter/text misalignment in the source metadata.
"""

from __future__ import annotations

import unittest

from src.evaluation.error_analysis import identify_errors
from src.evaluation.metrics import accuracy_by_field, answer_matches, exact_match_accuracy, extract_option_label, normalize_answer


class OptionLabelExtractionTests(unittest.TestCase):
    def test_extracts_leading_letter_with_period(self) -> None:
        self.assertEqual(extract_option_label("C. He is cooking in a kitchen"), "C")

    def test_extracts_letter_from_free_text_response(self) -> None:
        self.assertEqual(extract_option_label("The correct answer is C."), "C")

    def test_returns_none_when_no_option_letter_present(self) -> None:
        self.assertIsNone(extract_option_label("a dog is barking loudly"))


class AnswerMatchingTests(unittest.TestCase):
    def test_exact_letter_match(self) -> None:
        self.assertTrue(answer_matches("C. dog barking", "C. dog barking"))

    def test_free_form_response_matched_via_option_label(self) -> None:
        self.assertTrue(answer_matches("I believe the answer is C.", "C. dog barking"))

    def test_trailing_comma_artifact_does_not_break_matching(self) -> None:
        # Real record dev_aqa_1106: choice text ends with a stray "," that
        # the answer field does not have.
        choices = ["A. There is construction nearby,", "B. He is fixing a machine,", "C. He is cooking in a kitchen,", "D. It's part of the music"]
        reference = "C. He is cooking in a kitchen"
        prediction = "C. He is cooking in a kitchen,"
        self.assertTrue(answer_matches(prediction, reference, choices))

    def test_wrong_answer_does_not_match(self) -> None:
        self.assertFalse(answer_matches("A. cat meowing", "C. dog barking"))

    def test_letter_text_misalignment_case_dev_aqa_614(self) -> None:
        # Real record: answer field says "B. A countryside railway" but that
        # TEXT actually belongs to choice C ("B. A busy city street" is the
        # real choice B). A model correctly identifying the audio content
        # and answering "C. A countryside railway" should NOT be counted
        # wrong just because of this source-metadata defect on the label
        # digit -- current behavior is documented, not silently "fixed" by
        # guessing what the source dataset meant.
        choices = ["A. An airport runway", "B. A busy city street", "C. A countryside railway", "D. A quiet residential area"]
        reference = "B. A countryside railway"  # defective: text belongs to C
        prediction = "C. A countryside railway"  # semantically correct
        # Documented known limitation: this currently scores as NOT matching
        # because both the label (B vs C) and the raw text (differing
        # leading letter) disagree. See docs/dataset.md "Known ground-truth
        # defects" for the record_ids affected and the recommendation to
        # exclude them from strict accuracy reporting.
        self.assertFalse(answer_matches(prediction, reference, choices))


class ExactMatchAccuracyTests(unittest.TestCase):
    def test_accuracy_and_counts(self) -> None:
        result = exact_match_accuracy(["A. dog", "B. cat"], ["A. dog", "A. dog"])
        self.assertEqual(result["correct_count"], 1)
        self.assertEqual(result["incorrect_count"], 1)
        self.assertAlmostEqual(result["accuracy"], 0.5)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            exact_match_accuracy(["A"], ["A", "B"])


class PredictionRecordSchemaTests(unittest.TestCase):
    """The exact dict shape scripts/run_inference.py writes to
    results/predictions/predictions.jsonl must be consumable by every
    downstream evaluation/error-analysis function without KeyErrors.
    """

    def _sample_prediction_records(self) -> list[dict]:
        return [
            {
                "record_id": "dev_aqa_0",
                "audio_id": "dev_audio_00553",
                "audio_path": "dataset/dcase_part1/audio/dev/audio_00553.wav",
                "question": "What element in the audio contributes to the emotional depth?",
                "choices": ["A. The language spoken", "B. The steady drum beats", "C. The groovy bass line", "D. The keyboard harmony"],
                "expected_answer": "D. The keyboard harmony",
                "predicted_answer": "D. The keyboard harmony",
                "model_response": "D. The keyboard harmony",
                "question_type": "both",
                "source_split": "dev",
                "split": "test",
            },
            {
                "record_id": "dev_aqa_1",
                "audio_id": "dev_audio_00554",
                "audio_path": "dataset/dcase_part1/audio/dev/audio_00554.wav",
                "question": "How many distinct sounds are present?",
                "choices": ["A. 1", "B. 2", "C. 3", "D. 4"],
                "expected_answer": "B. 2",
                "predicted_answer": "C. 3",
                "model_response": "I think there are C. 3 sounds.",
                "question_type": "sound counting",
                "source_split": "dev",
                "split": "test",
            },
        ]

    def test_exact_match_accuracy_consumes_prediction_records(self) -> None:
        records = self._sample_prediction_records()
        result = exact_match_accuracy(
            [r["predicted_answer"] for r in records],
            [r["expected_answer"] for r in records],
            [r["choices"] for r in records],
        )
        self.assertEqual(result["correct_count"], 1)

    def test_accuracy_by_question_type(self) -> None:
        records = self._sample_prediction_records()
        by_type = accuracy_by_field(records, "question_type")
        self.assertIn("both", by_type)
        self.assertIn("sound counting", by_type)
        self.assertEqual(by_type["both"]["accuracy"], 1.0)
        self.assertEqual(by_type["sound counting"]["accuracy"], 0.0)

    def test_identify_errors_finds_the_wrong_record_only(self) -> None:
        records = self._sample_prediction_records()
        errors = identify_errors(records)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["record_id"], "dev_aqa_1")
        self.assertIn("analysis_category", errors[0])


if __name__ == "__main__":
    unittest.main()
