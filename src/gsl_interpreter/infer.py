from __future__ import annotations

import cv2
import joblib
import numpy as np
import torch

from gsl_interpreter import FEATURE_VERSION, NUM_FRAMES
from gsl_interpreter.camera import Camera
from gsl_interpreter.features import add_motion_features
from gsl_interpreter.labels import invert_labels
from gsl_interpreter.overlay import draw_text
from gsl_interpreter.smoothing import PredictionSmoother
from gsl_interpreter.torch_model import best_device, build_model

WINDOW_NAME = "GSL Interpreter"
RESET_BUTTON = (20, 105, 170, 155)
UNDO_BUTTON = (190, 105, 340, 155)
YELLOW = (0, 255, 255)
BACKGROUND_LABELS = {"არაფერი"}
BACKSPACE_LABELS = {"backspace", "delete", "undo", "წაშლა", "უკან"}
CLEAR_LABELS = {"clear", "reset", "გასუფთავება", "გასუფთავე"}
PUNCTUATION_LABELS = {
    ".": ".",
    "period": ".",
    "წერტილი": ".",
    ",": ",",
    "comma": ",",
    "მძიმე": ",",
    "?": "?",
    "question mark": "?",
    "კითხვის ნიშანი": "?",
    "!": "!",
    "exclamation mark": "!",
    "ძახილის ნიშანი": "!",
}
MIN_MEAN_STEP_MOTION = 0.12
MIN_START_END_MOTION = 0.65
ACTIVE_STEP_MOTION = 0.08
MIN_ACTIVE_STEPS = 6
TRIGGER_STEP_MOTION = 0.18
TRIGGER_CONSECUTIVE_FRAMES = 3
START_POSE_THRESHOLD = 4.5
POST_ACCEPT_COOLDOWN_FRAMES = 18
FEEDBACK_FRAMES = 28


def run_inference(model_path: str, camera_index: int = 0) -> None:
    from gsl_interpreter.landmarks import extract

    bundle = joblib.load(model_path)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(
            f"Model feature version {bundle.get('feature_version')} does not match {FEATURE_VERSION}"
        )

    labels_by_id = invert_labels(bundle["label_map"])
    allowed_labels = set(labels_by_id.values())
    start_templates = bundle.get("start_templates", {})
    device = best_device()
    model = _load_torch_model(bundle, len(labels_by_id), device)
    smoother = PredictionSmoother(threshold=0.75, consecutive=1)
    sequence: list[np.ndarray] = []
    previous_vector: np.ndarray | None = None
    motion_streak = 0
    capturing = False
    armed_label = ""
    cooldown_frames = 0
    feedback_text = ""
    feedback_frames = 0
    sentence_words: list[str] = []
    reset_requested = False
    undo_requested = False

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal reset_requested, undo_requested
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if _point_in_button(x, y, RESET_BUTTON):
            reset_requested = True
        elif _point_in_button(x, y, UNDO_BUTTON):
            undo_requested = True

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    with Camera(camera_index) as camera:
        for frame in camera.frames():
            if reset_requested:
                sentence_words.clear()
                capturing = False
                armed_label = ""
                cooldown_frames = 0
                motion_streak = 0
                previous_vector = None
                reset_requested = False
                feedback_text = "Cleared"
                feedback_frames = FEEDBACK_FRAMES
                smoother.reset()
                sequence.clear()

            if undo_requested:
                removed = _undo_sentence_word(sentence_words)
                capturing = False
                armed_label = ""
                cooldown_frames = 0
                motion_streak = 0
                previous_vector = None
                undo_requested = False
                feedback_text = f"Removed: {removed}" if removed else "Sentence empty"
                feedback_frames = FEEDBACK_FRAMES
                smoother.reset()
                sequence.clear()

            vector = extract(frame)
            if cooldown_frames > 0:
                cooldown_frames -= 1
                previous_vector = None
                motion_streak = 0
                armed_label = ""
            elif vector is None:
                previous_vector = None
                motion_streak = 0
                if not capturing:
                    armed_label = ""
            elif capturing:
                sequence.append(vector)
            else:
                if not armed_label:
                    armed_label = _match_start_pose(vector, start_templates)
                    previous_vector = vector if armed_label else None
                    motion_streak = 0
                elif previous_vector is not None:
                    step_motion = float(np.linalg.norm(vector - previous_vector))
                    if step_motion >= TRIGGER_STEP_MOTION:
                        motion_streak += 1
                    else:
                        motion_streak = 0

                    if motion_streak >= TRIGGER_CONSECUTIVE_FRAMES:
                        capturing = True
                        sequence = [previous_vector, vector]

                previous_vector = vector

            if capturing:
                status = f"Capturing {len(sequence)}/{NUM_FRAMES}"
                if len(sequence) >= NUM_FRAMES:
                    prediction = _predict_sequence(
                        sequence,
                        model,
                        labels_by_id,
                        allowed_labels,
                        smoother,
                        armed_label,
                    )
                    if prediction and prediction not in BACKGROUND_LABELS:
                        feedback_text = _apply_sentence_label(sentence_words, prediction)
                        feedback_frames = FEEDBACK_FRAMES
                        print(_sentence_text(sentence_words))
                    else:
                        feedback_text = "Try again"
                        feedback_frames = FEEDBACK_FRAMES
                    status = feedback_text
                    cooldown_frames = POST_ACCEPT_COOLDOWN_FRAMES
                    capturing = False
                    armed_label = ""
                    motion_streak = 0
                    sequence.clear()
                    previous_vector = None
                    smoother.reset()
            elif feedback_frames > 0:
                status = feedback_text
                feedback_frames -= 1
            else:
                status = "Move" if armed_label else "Ready"

            display = frame.copy()
            display = _draw_sentence_overlay(display, sentence_words, status)
            display = _draw_buttons(display)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in {ord("r"), ord("R")}:
                reset_requested = True
            if key in {8, 127, ord("u"), ord("U")}:
                undo_requested = True

    cv2.destroyAllWindows()


def _draw_buttons(frame: np.ndarray) -> np.ndarray:
    frame = _draw_button(frame, RESET_BUTTON, "Reset")
    return _draw_button(frame, UNDO_BUTTON, "Undo")


def _draw_button(frame: np.ndarray, rect: tuple[int, int, int, int], label: str) -> np.ndarray:
    left, top, right, bottom = rect
    cv2.rectangle(frame, (left, top), (right, bottom), YELLOW, 2)
    return draw_text(frame, label, (left + 20, top + 8), YELLOW, size=28)


def _draw_sentence_overlay(frame: np.ndarray, sentence_words: list[str], status: str) -> np.ndarray:
    sentence = _sentence_text(sentence_words) or "..."
    sentence = _shorten_text(sentence, max(18, frame.shape[1] // 17))
    frame = draw_text(frame, f"Sentence: {sentence}", (20, 25), YELLOW, size=34)
    return draw_text(frame, f"Status: {status}", (20, 68), YELLOW, size=28)


def _point_in_button(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom


def _apply_sentence_label(sentence_words: list[str], label: str) -> str:
    normalized = label.strip().casefold()

    if normalized in CLEAR_LABELS:
        sentence_words.clear()
        return "Cleared"

    if normalized in BACKSPACE_LABELS:
        removed = _undo_sentence_word(sentence_words)
        return f"Removed: {removed}" if removed else "Sentence empty"

    punctuation = PUNCTUATION_LABELS.get(normalized)
    if punctuation:
        if sentence_words:
            sentence_words[-1] = sentence_words[-1].rstrip(".,?!") + punctuation
        else:
            sentence_words.append(punctuation)
        return f"Added: {punctuation}"

    sentence_words.append(label)
    return f"Added: {label}"


def _undo_sentence_word(sentence_words: list[str]) -> str:
    if not sentence_words:
        return ""
    return sentence_words.pop()


def _sentence_text(sentence_words: list[str]) -> str:
    return " ".join(sentence_words).strip()


def _shorten_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[-max_chars:]
    return "..." + text[-(max_chars - 3) :]


def _predict_sequence(
    sequence: list[np.ndarray],
    model: torch.nn.Module,
    labels_by_id: dict[int, str],
    allowed_labels: set[str],
    smoother: PredictionSmoother,
    expected_label: str,
) -> str:
    if not _has_dynamic_motion(sequence):
        smoother.reset()
        return ""

    device = next(model.parameters()).device
    x_features = add_motion_features(np.stack(sequence).astype(np.float32))
    x = torch.from_numpy(x_features).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id])
    prediction = labels_by_id[class_id]
    if prediction not in allowed_labels:
        smoother.reset()
        return ""
    if expected_label and prediction != expected_label:
        smoother.reset()
        return ""
    return smoother.update(prediction, confidence) or ""


def _load_torch_model(
    bundle: dict[str, object],
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    if bundle.get("model_type") != "torch_sequence":
        raise RuntimeError("Model artifact is not a torch_sequence model. Retrain first.")
    config = bundle["model_config"]
    if not isinstance(config, dict):
        raise RuntimeError("Model artifact is missing model_config.")
    model = build_model(config, num_classes=num_classes).to(device)
    state_dict = bundle["state_dict"]
    if not isinstance(state_dict, dict):
        raise RuntimeError("Model artifact is missing state_dict.")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Using device: {device}")
    return model


def _has_dynamic_motion(sequence: list[np.ndarray]) -> bool:
    frames = np.stack(sequence).astype(np.float32)
    step_motion = np.linalg.norm(np.diff(frames, axis=0), axis=1)
    mean_step_motion = float(step_motion.mean())
    start_end_motion = float(np.linalg.norm(frames[-1] - frames[0]))
    active_steps = int(np.count_nonzero(step_motion >= ACTIVE_STEP_MOTION))
    return (
        mean_step_motion >= MIN_MEAN_STEP_MOTION
        and start_end_motion >= MIN_START_END_MOTION
        and active_steps >= MIN_ACTIVE_STEPS
    )


def _match_start_pose(
    vector: np.ndarray,
    start_templates: dict[str, np.ndarray],
) -> str:
    best_label = ""
    best_distance = float("inf")

    for label, templates in start_templates.items():
        if len(templates) == 0:
            continue
        distances = np.linalg.norm(templates - vector, axis=1)
        distance = float(np.min(distances))
        if distance < best_distance:
            best_distance = distance
            best_label = label

    if best_distance <= START_POSE_THRESHOLD:
        return best_label
    return ""
