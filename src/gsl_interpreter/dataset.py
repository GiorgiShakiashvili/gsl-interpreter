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
RECORDER_WINDOW = "GSL Recorder"
SAVE_BUTTON = (20, 118, 190, 176)
DISCARD_BUTTON = (210, 118, 420, 176)
PANEL_BG = (18, 20, 23)
TEXT = (240, 242, 245)
MUTED_TEXT = (160, 168, 176)
SUCCESS = (90, 190, 120)
DANGER = (80, 90, 225)


def record_samples(label: str, samples: int, signer: str, camera_index: int = 0) -> None:
    from gsl_interpreter.camera import Camera

    _validate_path_part(label, "label")
    _validate_path_part(signer, "signer")

    saved = 0
    cv2.namedWindow(RECORDER_WINDOW)
    with Camera(camera_index) as camera:
        frames = camera.frames()
        while saved < samples:
            _countdown(frames, label, saved + 1, samples)
            sample = _capture_sequence(frames)

            if sample is None:
                print("Capture aborted: too many empty hand-detection frames. Try again.")
                continue

            if _confirm_save(frames, sample, label, saved + 1, samples):
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
        cv2.imshow(RECORDER_WINDOW, display)
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
            cv2.imshow(RECORDER_WINDOW, display)
            if cv2.waitKey(1) == 27:
                raise KeyboardInterrupt


def _confirm_save(frames: object, sample: np.ndarray, label: str, index: int, total: int) -> bool:
    print(f"Captured sample shape={sample.shape}, dtype={sample.dtype}")
    decision: bool | None = None

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal decision
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if _point_in_rect(x, y, SAVE_BUTTON):
            decision = True
        elif _point_in_rect(x, y, DISCARD_BUTTON):
            decision = False

    cv2.setMouseCallback(RECORDER_WINDOW, on_mouse)
    while decision is None:
        frame = next(frames)
        display = draw_tracking(frame)
        display = _draw_confirmation_overlay(display, label, index, total)
        cv2.imshow(RECORDER_WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in {13, ord("y"), ord("Y"), ord("s"), ord("S")}:
            decision = True
        elif key in {27, ord("n"), ord("N"), ord("d"), ord("D")}:
            decision = False

    cv2.setMouseCallback(RECORDER_WINDOW, lambda *_args: None)
    return decision


def _draw_confirmation_overlay(frame: np.ndarray, label: str, index: int, total: int) -> np.ndarray:
    panel = (14, 14, min(frame.shape[1] - 14, 460), 196)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), PANEL_BG, -1)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (70, 76, 84), 1)
    frame = draw_text(frame, f"Sample {index}/{total}", (28, 28), TEXT, size=28)
    frame = draw_text(frame, f"Label: {label}", (28, 66), MUTED_TEXT, size=22)
    frame = draw_text(frame, "Save captured sample?", (28, 92), TEXT, size=20)
    frame = _draw_decision_button(frame, SAVE_BUTTON, "Save", SUCCESS)
    return _draw_decision_button(frame, DISCARD_BUTTON, "Discard", DANGER)


def _draw_decision_button(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (36, 42, 48), -1)
    cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color, 3)
    return draw_text(frame, label, (rect[0] + 24, rect[1] + 14), TEXT, size=26)


def _point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


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
