"""Verifies the P0-1 fix: supervised labels must mask padding and prompt
tokens, keeping only the assistant answer span as real training targets.

Uses a fake, dependency-free processor rather than the real Qwen2-Audio
processor (which requires downloading model weights) -- the fake mimics the
exact interface (apply_chat_template + __call__ returning input_ids /
attention_mask) and, crucially, the same structural property real chat
templates have: add_generation_prompt=True on the user-only conversation
produces a string that is an exact prefix of the full user+assistant
conversation's templated string. That is the property the collator's
prompt-length-masking logic depends on.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.collator import Qwen2AudioQACollator


class FakeTokenizer:
    def __init__(self) -> None:
        self.padding_side = "left"  # deliberately wrong; collator must force "right"
        self.vocab = {"<pad>": 0}

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def encode_words(self, text: str) -> list[int]:
        return [self._id(word) for word in text.split()]


class FakeFeatureExtractor:
    sampling_rate = 16000


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.feature_extractor = FakeFeatureExtractor()

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        parts: list[str] = []
        for turn in conversation:
            if turn["role"] == "user":
                for item in turn["content"]:
                    if item["type"] == "audio":
                        parts.append("<AUDIO>")
                    elif item["type"] == "text":
                        parts.append(item["text"])
            elif turn["role"] == "assistant":
                parts.append("<assistant>")
                parts.append(str(turn["content"]))
                parts.append("<end>")
        text = " ".join(" ".join(p.split()) for p in parts)
        if add_generation_prompt:
            text = (text + " <assistant>").strip()
        return text

    def __call__(self, text, audios=None, audio=None, sampling_rate=None, return_tensors="pt", padding=True):
        sequences = [self.tokenizer.encode_words(t) for t in text]
        max_len = max(len(seq) for seq in sequences)
        pad_id = self.tokenizer.vocab["<pad>"]
        input_ids, attention_mask = [], []
        for seq in sequences:
            pad_len = max_len - len(seq)
            if self.tokenizer.padding_side == "right":
                input_ids.append(seq + [pad_id] * pad_len)
                attention_mask.append([1] * len(seq) + [0] * pad_len)
            else:
                input_ids.append([pad_id] * pad_len + seq)
                attention_mask.append([0] * pad_len + [1] * len(seq))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class LabelMaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = FakeProcessor()
        self.collator = Qwen2AudioQACollator(processor=self.processor, sampling_rate=16000)
        patcher = patch("src.data.collator.load_audio_for_model", return_value=np.zeros(1600, dtype=np.float32))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_padding_side_is_forced_to_right(self) -> None:
        self.assertEqual(self.processor.tokenizer.padding_side, "right")

    def test_padding_positions_are_masked(self) -> None:
        features = [
            {"audio_path": "a.wav", "question": "What sound is this", "choices": ["A. dog", "B. cat"], "answer": "A. dog"},
            {"audio_path": "b.wav", "question": "What sound is this exact clip precisely", "choices": ["A. dog", "B. cat"], "answer": "B. cat"},
        ]
        batch = self.collator(features)
        labels, attention_mask = batch["labels"], batch["attention_mask"]
        self.assertTrue(torch.all(labels[attention_mask == 0] == -100))

    def test_prompt_tokens_are_masked_and_answer_tokens_survive(self) -> None:
        features = [
            {"audio_path": "a.wav", "question": "What sound is this", "choices": ["A. dog", "B. cat"], "answer": "A. dog"},
            {"audio_path": "b.wav", "question": "What sound is this exact clip precisely", "choices": ["A. dog", "B. cat"], "answer": "B. cat"},
        ]
        batch = self.collator(features)
        labels, attention_mask, input_ids = batch["labels"], batch["attention_mask"], batch["input_ids"]

        # No row should be entirely masked.
        self.assertTrue((labels != -100).any(dim=1).all().item())

        for row in range(labels.shape[0]):
            # First token (the <AUDIO> placeholder) is always part of the prompt.
            self.assertEqual(labels[row, 0].item(), -100)
            # The last real (non-padded) token is the answer's closing token
            # and must be a real label matching input_ids there.
            real_len = int(attention_mask[row].sum().item())
            self.assertNotEqual(labels[row, real_len - 1].item(), -100)
            self.assertEqual(labels[row, real_len - 1].item(), input_ids[row, real_len - 1].item())

    def test_train_on_prompt_true_disables_prompt_masking(self) -> None:
        collator = Qwen2AudioQACollator(processor=self.processor, sampling_rate=16000, train_on_prompt=True)
        features = [{"audio_path": "a.wav", "question": "What sound is this", "choices": None, "answer": "A dog barking"}]
        batch = collator(features)
        labels, attention_mask, input_ids = batch["labels"], batch["attention_mask"], batch["input_ids"]
        real_len = int(attention_mask[0].sum().item())
        self.assertTrue(torch.equal(labels[0, :real_len], input_ids[0, :real_len]))
        self.assertTrue(torch.all(labels[0, real_len:] == -100))


if __name__ == "__main__":
    unittest.main()
