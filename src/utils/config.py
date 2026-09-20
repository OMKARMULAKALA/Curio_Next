from __future__ import annotations

from pathlib import Path

import yaml

from src.utils.paths import resolve_project_path


def load_yaml(path: str | Path) -> dict:
    with resolve_project_path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def save_json(data: dict, path: str | Path) -> None:
    import json

    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")

