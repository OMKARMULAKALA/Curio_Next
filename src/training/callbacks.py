from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerCallback


class JsonlLogCallback(TrainerCallback):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            payload = {"step": state.global_step, "epoch": state.epoch, **logs}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

