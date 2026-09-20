from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.preprocessing import (
    build_training_conversation,
    build_user_conversation,
    format_multiple_choice_question,
)
from src.utils.audio import load_audio_for_model
from src.utils.paths import resolve_project_path

# Re-exported for backward compatibility: prompt construction now lives in
# src/data/preprocessing.py so both the training collator and inference
# (src/models/qwen2_audio.py) use the exact same canonical formatter.
__all__ = [
    "format_multiple_choice_question",
    "build_user_conversation",
    "build_training_conversation",
    "Qwen2AudioQACollator",
]


@dataclass
class Qwen2AudioQACollator:
    processor: object
    sampling_rate: int | None = None
    train_on_prompt: bool = False
    max_audio_seconds: float | None = None
    processor_audio_argument: str = "audios"
    metadata_keys: tuple[str, ...] = field(
        default=("record_id", "audio_id", "audio_path", "question", "choices", "answer", "question_type", "split")
    )

    def __post_init__(self) -> None:
        if self.sampling_rate is None:
            self.sampling_rate = int(getattr(self.processor.feature_extractor, "sampling_rate", 16000))
        # Prompt-length masking below assumes the real content of each
        # sequence starts at position 0 (i.e. padding is appended, not
        # prepended). Force right-padding explicitly rather than trusting
        # whatever the tokenizer's default happens to be, since left-padding
        # would silently break the label-masking offset.
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", "right") != "right":
            tokenizer.padding_side = "right"

    def _processor_call(self, text: list[str], audios: list[np.ndarray]):
        kwargs = {
            "text": text,
            self.processor_audio_argument: audios,
            "return_tensors": "pt",
            "padding": True,
        }
        try:
            return self.processor(**kwargs)
        except TypeError:
            if self.processor_audio_argument == "audios":
                kwargs.pop("audios")
                kwargs["audio"] = audios
                return self.processor(**kwargs)
            raise

    def __call__(self, features: list[dict]) -> dict:
        if not features:
            raise ValueError("Qwen2AudioQACollator received an empty feature list")
        full_texts: list[str] = []
        prompt_texts: list[str] = []
        audios: list[np.ndarray] = []
        metadata: list[dict] = []
        for item in features:
            audio = load_audio_for_model(
                resolve_project_path(item["audio_path"]),
                target_sampling_rate=int(self.sampling_rate),
                mono=True,
                max_seconds=self.max_audio_seconds,
            )
            prompt_conversation = build_user_conversation(item["question"], item.get("choices"), item["audio_path"])
            full_conversation = build_training_conversation(
                item["question"], item.get("choices"), item["audio_path"], item["answer"]
            )
            prompt_texts.append(
                self.processor.apply_chat_template(prompt_conversation, tokenize=False, add_generation_prompt=True)
            )
            full_texts.append(
                self.processor.apply_chat_template(full_conversation, tokenize=False, add_generation_prompt=False)
            )
            audios.append(audio)
            metadata.append({key: item.get(key) for key in self.metadata_keys if key in item})

        batch = self._processor_call(full_texts, audios)
        batch["labels"] = batch["input_ids"].clone()
        if "attention_mask" in batch:
            batch["labels"][batch["attention_mask"] == 0] = -100

        if not self.train_on_prompt:
            prompt_batch = self._processor_call(prompt_texts, audios)
            prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
            for row_index, prompt_length in enumerate(prompt_lengths):
                batch["labels"][row_index, : int(prompt_length)] = -100

        if (batch["labels"] != -100).sum().item() == 0:
            raise ValueError("All labels are masked. Check prompt/answer formatting and tokenizer alignment.")
        batch["metadata"] = metadata
        return batch
