from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from gsl_interpreter import FEATURE_VERSION
from gsl_interpreter.dataset import load_sequence_dataset
from gsl_interpreter.features import add_motion_features
from gsl_interpreter.torch_model import best_device, build_model, default_config


def train_model(model_out: str) -> None:
    x, y, label_map = load_sequence_dataset()
    class_ids = sorted(label_map.values())
    if len(class_ids) < 2:
        raise RuntimeError("Need at least two labels/classes to train a classifier.")
    if class_ids != list(range(len(class_ids))):
        raise RuntimeError("Label IDs must be contiguous. Recreate data/labels.json.")

    device = best_device()
    config = default_config()
    model = build_model(config, num_classes=len(class_ids)).to(device)

    x_model = np.stack([add_motion_features(sample) for sample in x])
    x_tensor = torch.from_numpy(x_model).to(device)
    y_tensor = torch.from_numpy(y).long().to(device)
    class_counts = np.bincount(y, minlength=len(class_ids)).astype(np.float32)
    class_weights = torch.from_numpy(1.0 / np.maximum(class_counts, 1.0)).to(device)
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600)

    model.train()
    epochs = 800 if len(y) < 100 else 500
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_x, batch_y = _make_augmented_batch(x, y, device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        if epoch in {1, epochs} or epoch % 50 == 0:
            with torch.no_grad():
                eval_logits = model(x_tensor)
                accuracy = (eval_logits.argmax(dim=1) == y_tensor).float().mean().item()
            print(
                f"epoch={epoch:03d} loss={loss.item():.4f} "
                f"accuracy={accuracy:.3f} device={device}"
            )

    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)
        accuracy = (logits.argmax(dim=1) == y_tensor).float().mean().item()
    print(f"Final training accuracy={accuracy:.3f} device={device}")

    bundle = {
        "model_type": "torch_sequence",
        "model_config": config,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "label_map": label_map,
        "feature_version": FEATURE_VERSION,
        "start_templates": _load_start_templates(label_map),
    }

    path = Path(model_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"Saved model bundle to {path}")
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _load_start_templates(label_map: dict[str, int]) -> dict[str, np.ndarray]:
    from gsl_interpreter.dataset import RAW_DATA_DIR, _fit_sequence_length

    templates: dict[str, list[np.ndarray]] = {}
    for label in label_map:
        label_dir = RAW_DATA_DIR / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.glob("*.npy")):
            sample = _fit_sequence_length(np.load(path).astype(np.float32))
            templates.setdefault(label, []).append(sample[0])

    return {label: np.stack(frames).astype(np.float32) for label, frames in templates.items()}


def _make_augmented_batch(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    copies: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for _ in range(copies):
        for sample in x:
            transformed = _augment_sequence(sample)
            augmented.append(add_motion_features(transformed))
        targets.append(y)

    batch_x = torch.from_numpy(np.stack(augmented)).to(device)
    batch_y = torch.from_numpy(np.concatenate(targets)).long().to(device)
    return batch_x, batch_y


def _augment_sequence(sample: np.ndarray) -> np.ndarray:
    augmented = sample.copy()
    augmented += np.random.normal(0.0, 0.01, augmented.shape).astype(np.float32)

    if np.random.random() < 0.7:
        scale = np.random.uniform(0.95, 1.05)
        augmented *= np.float32(scale)

    if np.random.random() < 0.7:
        shift = np.random.randint(-2, 3)
        if shift > 0:
            augmented = np.concatenate(
                [np.repeat(augmented[:1], shift, axis=0), augmented[:-shift]],
                axis=0,
            )
        elif shift < 0:
            amount = abs(shift)
            augmented = np.concatenate(
                [augmented[amount:], np.repeat(augmented[-1:], amount, axis=0)],
                axis=0,
            )

    return augmented.astype(np.float32)
