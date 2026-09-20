# Methodology

The PoC fine-tunes `Qwen/Qwen2-Audio-7B-Instruct` for multiple-choice Audio
QA on DCASE 2025 Task 5 Part 1.

## Fine-Tuning Approach

The model is loaded from pretrained weights and adapted with PEFT LoRA
(`src/models/lora.py`). The base 7B model is not trained from scratch.

**LoRA scope**: `Qwen2AudioForConditionalGeneration` contains two submodels
-- `audio_tower` (a Whisper-style encoder) and `language_model` (a Qwen2
decoder). Both expose attention projections named `q_proj`/`k_proj`/`v_proj`
(the audio tower's fourth projection is `out_proj`; the language model's is
`o_proj`), so naive suffix-based module matching would silently attach LoRA
to the audio tower's attention too. `configs/training.yaml`'s
`lora.module_scope: language_model` restricts target-module resolution and
matching to the `language_model.*` subtree; `lora.module_scope: all` is an
explicit, documented opt-in to widen this. Configured target modules are
validated against the actual loaded model at runtime
(`src/models/lora.py:validate_target_modules`) and training fails with a
specific error naming the offending module if a target doesn't resolve
in-scope, rather than silently training zero adapters for it.

LoRA hyperparameters (rank 16, alpha 32, dropout 0.05) are configured in
`configs/training.yaml`.

## Supervised Objective

Training uses supervised fine-tuning on
`audio + question + choices -> target answer`. Concretely
(`src/data/collator.py:Qwen2AudioQACollator`):

1. The user turn (audio placeholder + the canonical multiple-choice prompt,
   `src/data/preprocessing.py:format_multiple_choice_question`) and the
   assistant turn (the target answer, e.g. `"C. dog barking"`) are rendered
   through the processor's chat template into one text sequence per example.
2. The SAME conversation is also rendered with `add_generation_prompt=True`
   and no assistant turn, to determine each example's real (unpadded) prompt
   length -- via `attention_mask.sum()`, which is robust to how much padding
   surrounds the content and requires no assumption about token offsets.
3. Labels start as a clone of `input_ids`, then:
   - padding positions (`attention_mask == 0`) are set to `-100`;
   - unless `train_on_prompt: true` is set, every position up to the
     computed prompt length is also set to `-100`.
4. The result: **only the assistant's answer tokens (and the turn-closing
   token) contribute to the loss.** The question, choices, audio placeholder
   tokens, and padding are excluded. `train_on_prompt` is available as an
   explicit, documented way to include the full sequence in the loss
   instead, for anyone who wants to compare that ablation -- it actually
   controls collator behavior (verified in `tests/test_collator_labels.py`).
5. Tokenizer padding is forced to `padding_side="right"` in the collator
   (some tokenizers default to left-padding for generation-oriented use,
   which would silently break the prompt-length-masking offset above).

## Gradient Checkpointing + LoRA

Gradient checkpointing on a model with a fully frozen base (everything
except the LoRA adapters) can silently fail to produce gradients for the
adapters when using PyTorch's reentrant checkpointing implementation, since
the checkpointed region's inputs may all have `requires_grad=False`. This
project addresses it with the two changes that combination requires:
`model.enable_input_require_grads()` after LoRA injection
(`scripts/train.py`), and `gradient_checkpointing_kwargs={"use_reentrant":
False}` passed to `TrainingArguments`
(`src/training/trainer.py:build_training_args`).

## Evaluation

Ground-truth answers are `"<Letter>. <text>"` strings (e.g. `"C. dog
barking"`), drawn verbatim from the DCASE metadata. Two things are reported:

1. **Primary metric** (`src/evaluation/metrics.py:answer_matches`, used by
   `exact_match_accuracy`): first tries to extract an option letter (`A`-`D`)
   from both the prediction and the reference; if both extract, the labels
   are compared directly. This correctly scores a free-form response like
   `"The correct answer is C."` against a reference of `"C. dog barking"`.
   If either side has no extractable label, it falls back to a normalized
   (lowercased, punctuation-collapsed) full-text comparison, and finally
   checks whether the predicted label's own choice text matches the
   reference. No semantic-similarity or fuzzy matching is used.
2. **Question-type and source-split breakdowns**
   (`src/evaluation/metrics.py:accuracy_by_field`) apply the identical
   matching function per group.

This same matching function must be applied identically to the zero-shot
baseline and the fine-tuned model -- both read predictions through
`src/evaluation/evaluate.py:evaluate_prediction_records`, so there is only
one implementation to keep in sync.

Known limitation: 4 test records have a genuine ground-truth defect (see
docs/dataset.md, "Answer / Choice Consistency") that can score a correct
prediction as wrong. They are not excluded automatically; report headline
accuracy both with and without them if precision matters for a specific
comparison.

## Training Loop

`eval_strategy`/`save_strategy` are both `epoch`; `load_best_model_at_end` +
`metric_for_best_model: eval_loss` (`greater_is_better: false`) means the
checkpoint actually persisted to `best_model_dir` after training is the
epoch with the lowest validation loss, not necessarily epoch 100. Early
stopping exists (`training.yaml: early_stopping.enabled`) but defaults to
`false` and never fires unless explicitly turned on, so the requested
100-epoch run is what actually happens by default.
