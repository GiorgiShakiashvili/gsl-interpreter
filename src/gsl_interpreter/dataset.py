from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np

from gsl_interpreter import FEATURE_SIZE, NUM_FRAMES
from gsl_interpreter.labels import ensure_label, load_labels
from gsl_interpreter.overlay import draw_text
from gsl_interpreter.tracking_overlay import draw_tracking

RAW_DATA_DIR = Path("data/raw")
MAX_EMPTY_FRAMES = int(NUM_FRAMES * 0.2)


def record_samples(label: str, samples: int, signer: str, camera_index: int = 0) -> None:
    from gsl_interpreter.camera import Camera

    _validate_path_part(label, "label")
    _validate_path_part(signer, "signer")

    saved = 0
    with Camera(camera_index) as camera:
        frames = camera.frames()
        while saved < samples:
            _countdown(frames, label, saved + 1, samples)
            sample = _capture_sequence(frames)

            if sample is None:
                print("Capture aborted: too many empty hand-detection frames. Try again.")
                continue

            _show_preview(sample)
            answer = input("Save this sample? [Y/n] ").strip().lower()
            if answer in {"", "y", "yes"}:
                ensure_label(label)
                path = _sample_path(label, signer)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, sample.astype(np.float32))
                saved += 1
                print(f"Saved {path}")
            else:
                print("Discarded sample.")

    cv2.destroyAllWindows()


def load_dataset() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    label_map = load_labels()
    if not label_map:
        raise RuntimeError("No labels found. Record samples before training.")

    features: list[np.ndarray] = []
    targets: list[int] = []

    for label, class_id in label_map.items():
        label_dir = RAW_DATA_DIR / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.glob("*.npy")):
            sample = np.load(path).astype(np.float32)
            _validate_sample_shape(sample, path)
            sample = _fit_sequence_length(sample)
            features.append(sample.reshape(NUM_FRAMES * FEATURE_SIZE))
            targets.append(class_id)

    if not features:
        raise RuntimeError("No .npy samples found under data/raw.")

    return np.stack(features), np.array(targets, dtype=np.int64), label_map


def load_sequence_dataset() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    label_map = load_labels()
    if not label_map:
        raise RuntimeError("No labels found. Record samples before training.")

    features: list[np.ndarray] = []
    targets: list[int] = []

    for label, class_id in label_map.items():
        label_dir = RAW_DATA_DIR / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.glob("*.npy")):
            sample = np.load(path).astype(np.float32)
            _validate_sample_shape(sample, path)
            features.append(_fit_sequence_length(sample))
            targets.append(class_id)

    if not features:
        raise RuntimeError("No .npy samples found under data/raw.")

    return np.stack(features), np.array(targets, dtype=np.int64), label_map


def _capture_sequence(frames: object) -> np.ndarray | None:
    from gsl_interpreter.landmarks import extract

    captured: list[np.ndarray] = []
    empty = 0

    while len(captured) < NUM_FRAMES:
        frame = next(frames)
        vector = extract(frame)

        display = draw_tracking(frame)
        display = draw_text(
            display,
            f"Capturing {len(captured) + 1}/{NUM_FRAMES} empty={empty}",
            (20, 25),
            (0, 255, 0),
        )
        cv2.imshow("GSL Recorder", display)
        cv2.waitKey(1)

        if vector is None:
            empty += 1
            if empty > MAX_EMPTY_FRAMES:
                return None
            continue

        captured.append(vector)

    return np.stack(captured).astype(np.float32)


def _countdown(frames: object, label: str, index: int, total: int) -> None:
    for count in range(3, 0, -1):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            frame = next(frames)
            display = draw_tracking(frame)
            display = draw_text(
                display,
                f"Sample {index}/{total} - get ready: {count}",
                (20, 25),
                (0, 255, 255),
            )
            display = draw_text(display, f"Label: {label}", (20, 70), (0, 255, 255))
            cv2.imshow("GSL Recorder", display)
            if cv2.waitKey(1) == 27:
                raise KeyboardInterrupt


def _show_preview(sample: np.ndarray) -> None:
    print(f"Captured sample shape={sample.shape}, dtype={sample.dtype}")


def _sample_path(label: str, signer: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    return RAW_DATA_DIR / label / f"{signer}_{timestamp}_{millis:03d}.npy"


def _fit_sequence_length(sample: np.ndarray) -> np.ndarray:
    if sample.shape == (NUM_FRAMES, FEATURE_SIZE):
        return sample.astype(np.float32)
    fitted = np.zeros((NUM_FRAMES, FEATURE_SIZE), dtype=np.float32)
    frames = min(NUM_FRAMES, sample.shape[0])
    width = min(FEATURE_SIZE, sample.shape[1])
    fitted[:frames, :width] = sample[:frames, :width]
    return fitted


def _validate_sample_shape(sample: np.ndarray, path: Path) -> None:
    if sample.ndim != 2 or sample.shape[1] != FEATURE_SIZE:
        raise RuntimeError(
            f"{path} has shape {sample.shape}, but this feature version expects "
            f"(*, {FEATURE_SIZE}). Delete old samples and record fresh data."
        )


def _validate_path_part(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")
    if re.search(r'[\\/:"*?<>|]', value):
        raise ValueError(f"{name} cannot contain Windows path separator or reserved characters")
