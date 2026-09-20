from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_training_curves(log_file: str | Path = "logs/training_log.jsonl", output_dir: str | Path = "results/plots") -> list[Path]:
    import matplotlib.pyplot as plt

    log_file = Path(log_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not log_file.exists():
        raise FileNotFoundError(f"Training log not found: {log_file}")
    logs = pd.read_json(log_file, lines=True)
    outputs: list[Path] = []
    for column, filename, ylabel in [
        ("loss", "training_loss.png", "Training loss"),
        ("eval_loss", "validation_loss.png", "Validation loss"),
        ("learning_rate", "learning_rate.png", "Learning rate"),
    ]:
        if column not in logs:
            continue
        frame = logs.dropna(subset=[column])
        if frame.empty:
            continue
        x = "epoch" if "epoch" in frame else "step"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(frame[x], frame[column], marker="o", linewidth=1)
        ax.set_xlabel(x)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs

