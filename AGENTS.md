# Agent Instructions

This repository is the Audio Context Layer project.

- Dataset sources outside this directory are external sources only.
- Do not initialize additional Git repositories.
- Do not fabricate dataset fields.
- Inspect real data before implementing loaders.
- Do not leak audio between train, validation, and test splits.
- Do not use test data for training.
- Keep reusable code in `src/`.
- Keep experiments in `notebooks/`.
- Keep configuration in `configs/`.
- Keep outputs in `results/`.
- Keep checkpoints in `checkpoints/`.
- Do not commit audio or model checkpoints.
- Do not hard-code machine-specific paths.
- Record seeds and experiment configurations.
- Never fabricate evaluation results.

