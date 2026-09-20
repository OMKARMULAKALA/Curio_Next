from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import json
from pathlib import Path

from src.data.dataset_loader import load_manifest, load_split
from src.models.lora import load_lora_adapters
from src.models.qwen2_audio import generate_answer, load_qwen2_audio
from src.utils.audio import load_audio_for_model
from src.utils.config import load_yaml
from src.utils.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Audio QA predictions with a base model or LoRA checkpoint.")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--model", default=None, help="Override model.yaml's model_name (base model to load).")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--manifest", default=None, help="Explicit manifest .jsonl path, overriding --split.")
    parser.add_argument("--adapter-dir", "--checkpoint", dest="adapter_dir", default=None, help="LoRA adapter/checkpoint directory. Omit to evaluate the base (zero-shot) model.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/predictions/predictions.jsonl")
    args = parser.parse_args()

    dataset_cfg = load_yaml(args.dataset_config)
    model_cfg = load_yaml(args.model_config)
    model_name = args.model or model_cfg["model_name"]
    model, processor = load_qwen2_audio(model_name, **model_cfg.get("from_pretrained_kwargs", {}))
    if args.adapter_dir:
        model = load_lora_adapters(model, args.adapter_dir)
    sampling_rate = int(getattr(processor.feature_extractor, "sampling_rate", model_cfg.get("sampling_rate", 16000)))
    if args.manifest:
        records = load_manifest(args.manifest, schema=dataset_cfg.get("schema"))
    else:
        records = load_split(args.split, dataset_cfg["dataset_root"], schema=dataset_cfg.get("schema"))
    if args.limit:
        records = records[: args.limit]

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            audio = load_audio_for_model(resolve_project_path(record.audio_path), sampling_rate, max_seconds=model_cfg.get("max_audio_seconds"))
            response = generate_answer(
                model,
                processor,
                audio,
                record.question,
                record.choices,
                sampling_rate=sampling_rate,
                max_new_tokens=int(model_cfg.get("generation", {}).get("max_new_tokens", 64)),
            )
            prediction = {
                "record_id": record.record_id,
                "audio_id": record.audio_id,
                "audio_path": record.audio_path,
                "question": record.question,
                "choices": record.choices,
                "expected_answer": record.answer,
                "predicted_answer": response,
                "model_response": response,
                "question_type": record.question_type,
                "source_split": record.source_split,
                "split": record.split,
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
