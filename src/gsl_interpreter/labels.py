from __future__ import annotations

import json
from pathlib import Path

LABELS_PATH = Path("data/labels.json")


def load_labels(path: Path = LABELS_PATH) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {str(label): int(index) for label, index in data.items()}


def save_labels(label_map: dict[str, int], path: Path = LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(label_map.items(), key=lambda item: item[1]))
    with path.open("w", encoding="utf-8") as file:
        json.dump(ordered, file, ensure_ascii=False, indent=2)
        file.write("\n")


def ensure_label(label: str, path: Path = LABELS_PATH) -> int:
    label_map = load_labels(path)
    if label not in label_map:
        label_map[label] = len(label_map)
        save_labels(label_map, path)
    return label_map[label]


def invert_labels(label_map: dict[str, int]) -> dict[int, str]:
    return {index: label for label, index in label_map.items()}
