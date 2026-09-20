from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import json

from src.data.collator import format_multiple_choice_question
from src.data.dataset_loader import load_split
from src.data.validation import validate_splits
from src.utils.audio import audio_info
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight checks that do not require model download.")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()
    dataset_cfg = load_yaml(args.dataset_config)
    model_cfg = load_yaml(args.model_config)
    report = validate_splits(dataset_cfg["dataset_root"])
    train = load_split("train", dataset_cfg["dataset_root"])
    sample = train[0]
    smoke = {
        "split_validation_ok": report["ok"],
        "train_records": len(train),
        "sample_audio_info": audio_info(sample.audio_path),
        "sample_prompt": format_multiple_choice_question(sample.question, sample.choices),
        "model_name": model_cfg["model_name"],
        "note": "Processor/collator/model-forward smoke tests require installed dependencies and local model weights.",
    }
    print(json.dumps(smoke, indent=2))
    if not report["ok"]:
        raise SystemExit("Split validation failed.")


if __name__ == "__main__":
    main()
