from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
from pathlib import Path

from src.data.collator import Qwen2AudioQACollator
from src.data.dataset_loader import AudioQADataset, load_split
from src.models.lora import apply_lora, parameter_report
from src.models.qwen2_audio import load_qwen2_audio
from src.training.trainer import build_trainer, build_training_args
from src.utils.config import load_yaml, save_json
from src.utils.reproducibility import experiment_record, require_cuda, runtime_environment
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-Audio with LoRA for Audio QA.")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Bypass the CUDA requirement. Not supported for real training; "
        "only useful to smoke-test config/data wiring without a GPU.",
    )
    args = parser.parse_args()

    dataset_cfg = load_yaml(args.dataset_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.training_config)
    exp_cfg = load_yaml(args.experiment_config)
    set_seed(int(train_cfg["seed"]))

    if args.allow_cpu:
        print("WARNING: --allow-cpu set; skipping the CUDA requirement check. "
              "This is only appropriate for CPU-only smoke tests, never for the real experiment.")
        env = runtime_environment()
    else:
        env = require_cuda(min_vram_gb=float(train_cfg.get("min_vram_gb", 0.0)))
    print("Runtime environment:")
    print(env)
    save_json(
        experiment_record(exp_cfg, dataset_cfg, model_cfg, train_cfg),
        Path(train_cfg["output_dir"]) / "runtime_config.json",
    )

    model, processor = load_qwen2_audio(model_cfg["model_name"], **model_cfg.get("from_pretrained_kwargs", {}))
    model = apply_lora(model, train_cfg["lora"])
    if train_cfg.get("gradient_checkpointing", True):
        # enable_input_require_grads() forces the (frozen) input embedding
        # output to require grad, which is what lets gradients flow back to
        # the LoRA adapters through a checkpointed region when every other
        # input to that region is frozen. Without this, gradient
        # checkpointing + LoRA + a fully frozen base model can silently
        # produce a loss tensor with no grad_fn. See build_training_args()
        # for the paired gradient_checkpointing_kwargs={"use_reentrant": False}.
        model.enable_input_require_grads()
    report = parameter_report(model)
    print(f"Total parameters: {report.total_parameters}")
    print(f"Trainable parameters: {report.trainable_parameters}")
    print(f"Trainable percent: {report.trainable_percent:.4f}%")

    train_records = load_split("train", dataset_cfg["dataset_root"], schema=dataset_cfg.get("schema"))
    val_records = load_split("validation", dataset_cfg["dataset_root"], schema=dataset_cfg.get("schema"))
    collator = Qwen2AudioQACollator(
        processor=processor,
        sampling_rate=model_cfg.get("sampling_rate"),
        max_audio_seconds=model_cfg.get("max_audio_seconds"),
        processor_audio_argument=model_cfg.get("processor_audio_argument", "audios"),
    )
    training_args = build_training_args(train_cfg)
    trainer = build_trainer(
        model=model,
        args=training_args,
        train_dataset=AudioQADataset(train_records),
        eval_dataset=AudioQADataset(val_records),
        data_collator=collator,
        config=train_cfg,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(train_cfg["best_model_dir"])
    processor.save_pretrained(train_cfg["best_model_dir"])
    trainer.save_state()


if __name__ == "__main__":
    main()
