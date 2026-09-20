# Experiments

## Experiment Record

- Experiment ID: `qwen2_audio_lora_dcase_part1_100epoch_seed42`
- Dataset: DCASE 2025 Task 5 Part 1 available local subset, leakage-remediated
  (see docs/dataset.md) — train 1,436 / validation 158 / test 2,466 records
- Model: `Qwen/Qwen2-Audio-7B-Instruct`
- Fine-tuning: PEFT LoRA, scoped to `language_model.*` (see docs/methodology.md)
- Epochs: 100 (`configs/training.yaml`)
- Batch size / learning rate / scheduler: see `configs/training.yaml`
- Seed: 42
- Hardware: recorded per run in `<output_dir>/runtime_config.json` via
  `src.utils.reproducibility.runtime_environment()` (written by `scripts/train.py`)

## Pipeline

1. `scripts/build_manifests.py` — manifests and splits
2. `scripts/validate_splits.py` — leakage checks
3. `scripts/validate_audio_integrity.py` — decodability
4. `scripts/train.py` — LoRA fine-tuning (100 epochs, best checkpoint by validation loss)
5. `scripts/run_inference.py` — test-split predictions
6. `scripts/evaluate.py` — metrics and error analysis
7. `scripts/generate_report.py` — refresh `docs/final_report.md` and the PDF from `results/`

## Results

Quantitative outputs live under `results/metrics/`, `results/predictions/`, and
`results/plots/`. Regenerate the final report after evaluation so Sections 21–24
reflect the latest run.
