from __future__ import annotations

from src.data.preprocessing import build_user_conversation


def load_qwen2_audio(model_name: str, **kwargs):
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, **kwargs)
    return model, processor


def move_batch_to_device(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def generate_answer(model, processor, audio, question: str, choices: list[str] | None = None, sampling_rate: int = 16000, max_new_tokens: int = 64) -> str:
    # Uses the SAME conversation builder as the training collator
    # (src/data/collator.py -> src/data/preprocessing.py) so the fine-tuned
    # model is prompted at inference exactly as it was trained.
    conversation = build_user_conversation(question, choices, audio_path="audio")
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    try:
        inputs = processor(text=[text], audios=[audio], return_tensors="pt", padding=True)
    except TypeError:
        inputs = processor(text=[text], audio=[audio], return_tensors="pt", padding=True)
    inputs = move_batch_to_device(inputs, model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
