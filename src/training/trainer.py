from __future__ import annotations

from pathlib import Path

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from src.training.callbacks import JsonlLogCallback


class AudioQATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        inputs.pop("metadata", None)
        if num_items_in_batch is not None:
            outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss


def build_training_args(config: dict) -> TrainingArguments:
    gradient_checkpointing = bool(config.get("gradient_checkpointing", True))
    kwargs = dict(
        output_dir=config["output_dir"],
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
        warmup_ratio=float(config.get("warmup_ratio", 0.0)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        optim=config.get("optim", "adamw_torch"),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        logging_steps=int(config.get("logging_steps", 10)),
        eval_strategy=config.get("eval_strategy", "epoch"),
        save_strategy="epoch",
        save_steps=config.get("save_steps"),
        load_best_model_at_end=True,
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(config.get("greater_is_better", False)),
        fp16=bool(config.get("fp16", False)),
        bf16=bool(config.get("bf16", False)),
        gradient_checkpointing=gradient_checkpointing,
        report_to=config.get("report_to", "none"),
        seed=int(config.get("seed", 42)),
        save_total_limit=int(config.get("save_total_limit", 3)),
        remove_unused_columns=False,
    )
    if gradient_checkpointing:
        # Required for LoRA (frozen base model) + gradient checkpointing to
        # actually produce gradients for the adapters: reentrant checkpointing
        # re-runs the forward pass under torch.no_grad() internally and can
        # silently break the graph when the checkpointed region's inputs all
        # have requires_grad=False, which is exactly the case for a frozen
        # base model's embedding output. use_reentrant=False (the
        # TorchDynamo-compatible implementation) avoids this. Combined with
        # model.enable_input_require_grads() in scripts/train.py.
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**kwargs)


def build_trainer(model, args: TrainingArguments, train_dataset, eval_dataset, data_collator, config: dict) -> Trainer:
    callbacks = [JsonlLogCallback(Path(config.get("log_file", "logs/training_log.jsonl")))]
    if config.get("early_stopping", {}).get("enabled", False):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(config["early_stopping"].get("patience", 5))))
    return AudioQATrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
