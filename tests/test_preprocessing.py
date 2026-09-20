from __future__ import annotations

import unittest

from src.data.preprocessing import (
    build_training_conversation,
    build_user_conversation,
    format_multiple_choice_question,
    normalize_raw_record,
)
from src.models.qwen2_audio import build_user_conversation as inference_build_user_conversation


class PromptFormattingTests(unittest.TestCase):
    def test_no_choices_returns_bare_question(self) -> None:
        self.assertEqual(format_multiple_choice_question("What is that sound?", None), "What is that sound?")

    def test_choices_are_included_verbatim(self) -> None:
        prompt = format_multiple_choice_question("What is that sound?", ["A. Dog", "B. Cat"])
        self.assertIn("A. Dog", prompt)
        self.assertIn("B. Cat", prompt)
        self.assertIn("What is that sound?", prompt)

    def test_training_and_inference_use_the_identical_user_prompt(self) -> None:
        # P0-7: training (build_user_conversation, used by the collator) and
        # inference (src.models.qwen2_audio.generate_answer) must format the
        # user turn identically. They now both import the SAME function, but
        # this test guards against someone re-introducing a second copy.
        question, choices, audio_path = "What is that sound?", ["A. Dog", "B. Cat"], "some/audio.wav"
        self.assertEqual(
            build_user_conversation(question, choices, audio_path),
            inference_build_user_conversation(question, choices, audio_path),
        )

    def test_training_conversation_appends_assistant_turn(self) -> None:
        conversation = build_training_conversation("Q", ["A. x", "B. y"], "a.wav", "A. x")
        self.assertEqual(conversation[-1]["role"], "assistant")
        self.assertEqual(conversation[-1]["content"], "A. x")


class NormalizeRawRecordTests(unittest.TestCase):
    def test_maps_raw_dcase_fields_to_manifest_fields(self) -> None:
        raw = {
            "question": "How many sounds?",
            "choice": ["A. 1", "B. 2"],
            "answer": "B. 2",
            "id": "fold1-a-0001",
            "audio_url": "../../local_audio_path/train/fold1-a-0001.wav",
            "question_type": "sound counting",
        }
        record = normalize_raw_record(raw, split="train", metadata_file="dataset/dcase_part1/metadata/train/fold1-a-0001.json", audio_root="dataset/dcase_part1/audio")
        self.assertEqual(record["audio_id"], "fold1-a-0001")
        self.assertEqual(record["audio_file"], "fold1-a-0001.wav")
        self.assertEqual(record["audio_path"], "dataset/dcase_part1/audio/train/fold1-a-0001.wav")
        self.assertEqual(record["choices"], ["A. 1", "B. 2"])
        self.assertEqual(record["answer"], "B. 2")
        self.assertEqual(record["source_split"], "train")
        self.assertNotIn("split", record)  # split is assigned later by the manifest builder


if __name__ == "__main__":
    unittest.main()
