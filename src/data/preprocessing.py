from __future__ import annotations

import json
from pathlib import Path

from src.utils.paths import resolve_project_path


def normalize_raw_record(raw: dict, split: str, metadata_file: str, audio_root: str | Path) -> dict:
    audio_file = Path(raw["audio_url"]).name
    audio_path = Path(audio_root) / split / audio_file
    return {
        "record_id": Path(metadata_file).stem,
        "audio_id": raw["id"],
        "audio_file": audio_file,
        "audio_path": str(audio_path).replace("\\", "/"),
        "question": raw["question"],
        "choices": raw.get("choice"),
        "answer": raw["answer"],
        "question_type": raw.get("question_type"),
        "source_split": split,
        "metadata_file": metadata_file.replace("\\", "/"),
        "original_audio_url": raw.get("audio_url"),
    }


def read_raw_json(path: str | Path) -> dict:
    with resolve_project_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Canonical multiple-choice prompt formatting.
#
# This is the ONLY place the question+choices prompt text and the
# user/assistant conversation shape should be constructed. Training
# (src/data/collator.py) and inference (src/models/qwen2_audio.py,
# scripts/run_inference.py) both import these functions rather than building
# their own prompt strings -- a prior implementation had two independently
# drifting copies of this logic, which meant the fine-tuned model would be
# evaluated on a differently-formatted prompt than it was trained on.
# ---------------------------------------------------------------------------


def format_multiple_choice_question(question: str, choices: list[str] | None) -> str:
    if not choices:
        return question.strip()
    choices_text = "\n".join(str(choice).strip() for choice in choices)
    return (
        f"{question.strip()}\n\n"
        f"Choices:\n{choices_text}\n\n"
        "Respond with the exact option letter and answer text, for example: B. answer text"
    )


def build_user_conversation(question: str, choices: list[str] | None, audio_path: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": format_multiple_choice_question(question, choices)},
            ],
        }
    ]


def build_training_conversation(question: str, choices: list[str] | None, audio_path: str, answer: str) -> list[dict]:
    conversation = build_user_conversation(question, choices, audio_path)
    conversation.append({"role": "assistant", "content": str(answer).strip()})
    return conversation
