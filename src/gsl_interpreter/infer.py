from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

from gsl_interpreter import FEATURE_VERSION, NUM_FRAMES
from gsl_interpreter.camera import Camera
from gsl_interpreter.labels import invert_labels
from gsl_interpreter.overlay import draw_text, draw_text_batch
from gsl_interpreter.smoothing import PredictionSmoother
from gsl_interpreter.torch_model import best_device, build_model
from gsl_interpreter.tts import DEFAULT_GEORGIAN_VOICE, GeorgianTTS, TTSQueueResult

WINDOW_NAME = "GSL Interpreter"
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
SPOKEN_SAVE_EVENTS = {"manual", "recording_stop", "session_end"}
MIN_MEAN_STEP_MOTION = 0.12
MIN_START_END_MOTION = 0.65
ACTIVE_STEP_MOTION = 0.08
MIN_ACTIVE_STEPS = 6
TRIGGER_STEP_MOTION = 0.18
TRIGGER_CONSECUTIVE_FRAMES = 3
START_POSE_THRESHOLD = 4.5
POST_ACCEPT_COOLDOWN_FRAMES = 18
FEEDBACK_FRAMES = 28
APP_BG = (16, 18, 21)
PANEL_BG = (25, 28, 32)
PANEL_BG_SOFT = (38, 43, 49)
BORDER = (67, 74, 82)
TEXT = (238, 241, 243)
MUTED_TEXT = (154, 164, 172)
SUBTLE_TEXT = (109, 119, 128)
ACCENT = (210, 143, 62)
SUCCESS = (124, 184, 92)
WARNING = (54, 172, 222)
DANGER = (72, 82, 212)
_HUD_PANEL_CACHE: dict[str, object] = {}


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float


@dataclass(frozen=True)
class SentenceSaveResult:
    saved: bool
    message: str
    path: str


class SentenceRecorder:
    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path else None

    def record(
        self,
        sentence_words: list[str],
        event: str,
        source_label: str = "",
        confidence: float | None = None,
    ) -> SentenceSaveResult:
        sentence = _sentence_text(sentence_words)
        if not sentence:
            return SentenceSaveResult(False, "Sentence empty", str(self.path or ""))
        if self.path is None:
            return SentenceSaveResult(False, "Sentence log disabled", "")

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "sentence": sentence,
            "words": sentence_words.copy(),
        }
        if source_label:
            record["source_label"] = source_label
        if confidence is not None:
            record["confidence"] = round(confidence, 6)

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(line + "\n")
                file.flush()
            return SentenceSaveResult(True, "Saved sentence", str(self.path))
        except OSError as exc:
            reason = exc.strerror or str(exc)
            return SentenceSaveResult(False, f"Save failed: {reason}", str(self.path))


def run_inference(
    model_path: str,
    camera_index: int = 0,
    width: int | None = 960,
    height: int | None = 540,
    camera_buffer: int = 1,
    start_threshold: float = START_POSE_THRESHOLD,
    tracking_complexity: int = 1,
    sentence_log: str | None = "data/sentences.jsonl",
    autosave_sentences: bool = False,
    tts_mode: str = "saved",
    tts_voice: str = DEFAULT_GEORGIAN_VOICE,
    tts_rate: str = "+0%",
    tts_volume: str = "+0%",
    tts_cache_dir: str = "data/tts",
) -> None:
    from gsl_interpreter import landmarks

    landmarks.configure(model_complexity=tracking_complexity)
    extract = landmarks.extract

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
    sentence_recorder = SentenceRecorder(sentence_log)
    sentence_tts = GeorgianTTS(
        mode=tts_mode,
        voice=tts_voice,
        rate=tts_rate,
        volume=tts_volume,
        cache_dir=tts_cache_dir,
    )
    sequence: list[np.ndarray] = []
    previous_vector: np.ndarray | None = None
    motion_streak = 0
    capturing = False
    armed_label = ""
    start_hint = ""
    cooldown_frames = 0
    feedback_text = ""
    feedback_frames = 0
    sentence_words: list[str] = []
    last_result: PredictionResult | None = None
    button_rects: dict[str, tuple[int, int, int, int]] = {}
    recording_active = True
    record_toggle_requested = False
    reset_requested = False
    undo_requested = False
    save_requested = False

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal record_toggle_requested, reset_requested, undo_requested, save_requested
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if _point_in_button(x, y, button_rects.get("record")):
            record_toggle_requested = True
        elif _point_in_button(x, y, button_rects.get("save")):
            save_requested = True
        elif _point_in_button(x, y, button_rects.get("reset")):
            reset_requested = True
        elif _point_in_button(x, y, button_rects.get("undo")):
            undo_requested = True

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    with Camera(camera_index, width=width, height=height, buffer_size=camera_buffer) as camera:
        for frame in camera.frames():
            if record_toggle_requested:
                if recording_active:
                    stop_save, tts_result = _record_and_maybe_speak(
                        sentence_recorder,
                        sentence_tts,
                        sentence_words,
                        event="recording_stop",
                    )
                    feedback_text = _with_tts_feedback(stop_save.message, tts_result)
                    recording_active = False
                else:
                    sentence_words.clear()
                    recording_active = True
                    feedback_text = "Recording started"
                    last_result = None
                capturing = False
                armed_label = ""
                start_hint = ""
                cooldown_frames = 0
                motion_streak = 0
                previous_vector = None
                record_toggle_requested = False
                feedback_frames = FEEDBACK_FRAMES
                smoother.reset()
                sequence.clear()

            if reset_requested:
                reset_save = sentence_recorder.record(sentence_words, event="reset") if sentence_words else None
                sentence_words.clear()
                capturing = False
                armed_label = ""
                start_hint = ""
                cooldown_frames = 0
                motion_streak = 0
                previous_vector = None
                reset_requested = False
                recording_active = True
                feedback_text = reset_save.message if reset_save and not reset_save.saved else "Cleared"
                feedback_frames = FEEDBACK_FRAMES
                last_result = None
                smoother.reset()
                sequence.clear()

            if save_requested:
                save_result, tts_result = _record_and_maybe_speak(
                    sentence_recorder,
                    sentence_tts,
                    sentence_words,
                    event="manual",
                )
                feedback_text = _with_tts_feedback(save_result.message, tts_result)
                feedback_frames = FEEDBACK_FRAMES
                save_requested = False

            if undo_requested:
                removed = _undo_sentence_word(sentence_words)
                capturing = False
                armed_label = ""
                start_hint = ""
                cooldown_frames = 0
                motion_streak = 0
                previous_vector = None
                undo_requested = False
                feedback_text = f"Removed: {removed}" if removed else "Sentence empty"
                feedback_frames = FEEDBACK_FRAMES
                last_result = None
                if autosave_sentences and removed:
                    undo_save = sentence_recorder.record(
                        sentence_words,
                        event="undo",
                        source_label=removed,
                    )
                    if not undo_save.saved and undo_save.message != "Sentence empty":
                        feedback_text = undo_save.message
                smoother.reset()
                sequence.clear()

            vector = extract(frame)
            start_hint = ""
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
                    start_label, start_distance = _best_start_pose_match(vector, start_templates)
                    if start_label and start_distance <= start_threshold:
                        armed_label = start_label
                    elif start_label:
                        start_hint = f"Move: {start_label} {start_distance:.1f}"

                if previous_vector is not None:
                    step_motion = float(np.linalg.norm(vector - previous_vector))
                    if step_motion >= TRIGGER_STEP_MOTION:
                        motion_streak += 1
                    else:
                        motion_streak = 0

                    if motion_streak >= TRIGGER_CONSECUTIVE_FRAMES:
                        capturing = True
                        sequence = [previous_vector, vector]
                else:
                    motion_streak = 0

                previous_vector = vector

            if capturing:
                status = f"Capturing {len(sequence)}/{NUM_FRAMES}"
                if len(sequence) >= NUM_FRAMES:
                    result = _predict_sequence(
                        sequence,
                        model,
                        labels_by_id,
                        allowed_labels,
                        smoother,
                        armed_label,
                    )
                    if result and result.label not in BACKGROUND_LABELS:
                        if recording_active:
                            feedback_text = _apply_sentence_label(sentence_words, result.label)
                            if _should_speak_label(result.label):
                                feedback_text = _with_tts_feedback(
                                    feedback_text,
                                    sentence_tts.speak_word(result.label),
                                )
                        else:
                            feedback_text = f"Detected: {result.label}"
                        feedback_frames = FEEDBACK_FRAMES
                        last_result = result
                        if autosave_sentences and recording_active:
                            update_save = sentence_recorder.record(
                                sentence_words,
                                event="update",
                                source_label=result.label,
                                confidence=result.confidence,
                            )
                            if not update_save.saved:
                                feedback_text = update_save.message
                        if recording_active:
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
                if recording_active:
                    status = "Move" if armed_label else start_hint or "Recording"
                else:
                    status = "Press Record"

            display = frame.copy()
            progress = len(sequence) / NUM_FRAMES if capturing else 0.0
            display, button_rects = _draw_inference_hud(
                display,
                sentence_words,
                status,
                progress,
                armed_label,
                last_result,
                recording_active,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in {ord("r"), ord("R")}:
                reset_requested = True
            if key in {8, 127, ord("u"), ord("U")}:
                undo_requested = True
            if key in {10, 13, ord("s"), ord("S")}:
                save_requested = True
            if key in {ord("t"), ord("T"), ord(" ")}:
                record_toggle_requested = True

    if recording_active and sentence_words:
        _record_and_maybe_speak(
            sentence_recorder,
            sentence_tts,
            sentence_words,
            event="session_end",
        )
    cv2.destroyAllWindows()


def _draw_inference_hud(
    frame: np.ndarray,
    sentence_words: list[str],
    status: str,
    progress: float,
    armed_label: str,
    last_result: PredictionResult | None,
    recording_active: bool,
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    height, width = frame.shape[:2]
    panel_width = max(300, min(390, int(width * 0.46)))
    panel, local_buttons = _cached_console_panel(
        height,
        panel_width,
        sentence_words,
        status,
        progress,
        armed_label,
        last_result,
        recording_active,
    )

    canvas = np.empty((height, width + panel_width, 3), dtype=np.uint8)
    canvas[:, :width] = frame
    canvas[:, width:] = panel
    _draw_camera_badge(canvas)
    buttons = {
        key: (rect[0] + width, rect[1], rect[2] + width, rect[3])
        for key, rect in local_buttons.items()
    }
    return canvas, buttons


def _cached_console_panel(
    height: int,
    panel_width: int,
    sentence_words: list[str],
    status: str,
    progress: float,
    armed_label: str,
    last_result: PredictionResult | None,
    recording_active: bool,
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    last_key = (
        last_result.label,
        round(last_result.confidence, 3),
    ) if last_result else None
    cache_key = (
        height,
        panel_width,
        tuple(sentence_words),
        status,
        round(progress, 2),
        armed_label,
        last_key,
        recording_active,
    )
    if _HUD_PANEL_CACHE.get("key") == cache_key:
        return (
            _HUD_PANEL_CACHE["panel"],  # type: ignore[return-value]
            _HUD_PANEL_CACHE["buttons"],  # type: ignore[return-value]
        )

    panel, buttons = _render_console_panel(
        height,
        panel_width,
        sentence_words,
        status,
        progress,
        armed_label,
        last_result,
        recording_active,
    )
    _HUD_PANEL_CACHE["key"] = cache_key
    _HUD_PANEL_CACHE["panel"] = panel
    _HUD_PANEL_CACHE["buttons"] = buttons
    return panel, buttons


def _render_console_panel(
    height: int,
    panel_width: int,
    sentence_words: list[str],
    status: str,
    progress: float,
    armed_label: str,
    last_result: PredictionResult | None,
    recording_active: bool,
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    compact = height < 420
    canvas = np.full((height, panel_width, 3), APP_BG, dtype=np.uint8)
    cv2.line(canvas, (0, 0), (0, height), BORDER, 1)
    text_items: list[tuple[str, tuple[int, int], tuple[int, int, int], int]] = []

    pad = 14 if compact else 20
    gap = 8 if compact else 12
    header_h = 48 if compact else 64
    status_h = 64 if compact else 88
    controls_h = 102 if compact else 128

    content_left = pad
    content_right = panel_width - pad
    header_top = pad
    header_bottom = header_top + header_h
    status_top = header_bottom + gap
    status_bottom = status_top + status_h
    controls_top = height - pad - controls_h
    transcript_top = status_bottom + gap
    transcript_bottom = max(transcript_top + 44, controls_top - gap)

    text_items.append(("GSL Interpreter", (content_left, header_top + 2), TEXT, 24 if not compact else 20))
    mode = "Recording active" if recording_active else "Recording paused"
    text_items.append((mode, (content_left, header_top + 33), MUTED_TEXT, 15 if not compact else 13))
    cv2.line(canvas, (content_left, header_bottom), (content_right, header_bottom), BORDER, 1)

    status_rect = (content_left, status_top, content_right, status_bottom)
    canvas = _draw_section(canvas, status_rect)
    status_label = _status_label(status, armed_label)
    status_color = _status_color(status)
    indicator_rect = (status_rect[0] + 14, status_rect[1] + 18, status_rect[0] + 20, status_rect[3] - 18)
    canvas = _draw_round_rect_alpha(canvas, indicator_rect, status_color, alpha=1.0, radius=3)
    text_items.append(("RECOGNITION", (status_rect[0] + 32, status_rect[1] + 11), SUBTLE_TEXT, 12))
    text_items.append(
        (
            _shorten_text(status_label, max(14, (status_rect[2] - status_rect[0]) // 12)),
            (status_rect[0] + 32, status_rect[1] + 32),
            TEXT,
            20 if not compact else 17,
        )
    )
    if progress > 0:
        bar_rect = (status_rect[0] + 32, status_rect[3] - 18, status_rect[2] - 14, status_rect[3] - 11)
        canvas = _draw_progress_bar(canvas, bar_rect, progress)

    transcript_rect = (content_left, transcript_top, content_right, transcript_bottom)
    canvas = _draw_section(canvas, transcript_rect)
    text_items.append(("TRANSCRIPT", (transcript_rect[0] + 14, transcript_rect[1] + 12), SUBTLE_TEXT, 12))

    sentence_text = _sentence_text(sentence_words) or "No sentence yet"
    max_chars = max(18, (transcript_rect[2] - transcript_rect[0] - 28) // (11 if compact else 13))
    max_lines = max(1, min(4, (transcript_rect[3] - transcript_rect[1] - 82) // (26 if compact else 32)))
    lines = _wrap_text(sentence_text, max_chars=max_chars, max_lines=max_lines)
    line_y = transcript_rect[1] + 38
    text_size = 20 if compact else 24
    for line in lines:
        text_items.append((line, (transcript_rect[0] + 14, line_y), TEXT, text_size))
        line_y += 28 if compact else 34

    if last_result:
        confidence = int(round(last_result.confidence * 100))
        meta = f"Last detection: {_shorten_text(last_result.label, 18)}  Confidence {confidence}%"
    else:
        meta = "Last detection: none"
    text_items.append((meta, (transcript_rect[0] + 14, transcript_rect[3] - 51), MUTED_TEXT, 14))

    chip_y = transcript_rect[3] - 26
    chip_x = transcript_rect[0] + 14
    for chip in sentence_words[-3 if compact else -4:]:
        chip_label = _shorten_text(chip, 14)
        chip_width = 22 + len(chip_label) * 9
        if chip_x + chip_width > transcript_rect[2] - 14:
            break
        chip_rect = (chip_x, chip_y, chip_x + chip_width, chip_y + 20)
        canvas = _draw_chip(canvas, chip_rect, chip_label, text_items)
        chip_x += chip_width + 6

    controls_rect = (content_left, controls_top, content_right, height - pad)
    canvas = _draw_section(canvas, controls_rect)
    text_items.append(("ACTIONS", (controls_rect[0] + 14, controls_rect[1] + 10), SUBTLE_TEXT, 12))
    buttons = _control_button_rects(controls_rect, compact=compact)
    record_label = "Stop" if recording_active else "Record"
    record_color = DANGER if recording_active else SUCCESS
    canvas = _draw_button(canvas, buttons["record"], record_label, record_color, text_items)
    canvas = _draw_button(canvas, buttons["save"], "Save", SUCCESS, text_items)
    canvas = _draw_button(canvas, buttons["undo"], "Undo", ACCENT, text_items)
    canvas = _draw_button(canvas, buttons["reset"], "Reset", DANGER, text_items)

    canvas = draw_text_batch(canvas, text_items)
    return canvas, buttons


def _draw_camera_badge(canvas: np.ndarray) -> None:
    rect = (16, 16, 142, 42)
    _draw_round_rect_alpha_in_place(canvas, rect, PANEL_BG, alpha=0.82, radius=6)
    _draw_round_rect(canvas, rect, BORDER, thickness=1, radius=6)
    cv2.putText(
        canvas,
        "CAMERA 01",
        (30, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT,
        1,
        cv2.LINE_AA,
    )


def _control_button_rects(
    panel_rect: tuple[int, int, int, int],
    compact: bool,
) -> dict[str, tuple[int, int, int, int]]:
    left, top, right, bottom = panel_rect
    panel_width = right - left
    gap = 8
    button_height = 34 if compact else 40
    first_row_top = top + 34
    second_row_top = min(bottom - button_height - 14, first_row_top + button_height + gap)
    button_width = (panel_width - 42 - gap) // 2
    left_button = left + 14
    right_button = left_button + button_width + gap
    return {
        "record": (left_button, first_row_top, left_button + button_width, first_row_top + button_height),
        "save": (right_button, first_row_top, right_button + button_width, first_row_top + button_height),
        "undo": (left_button, second_row_top, left_button + button_width, second_row_top + button_height),
        "reset": (right_button, second_row_top, right_button + button_width, second_row_top + button_height),
    }


def _draw_section(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    frame = _draw_round_rect_alpha(frame, rect, PANEL_BG, alpha=1.0, radius=8)
    _draw_round_rect(frame, rect, BORDER, thickness=1, radius=8)
    return frame


def _draw_button(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    text_items: list[tuple[str, tuple[int, int], tuple[int, int, int], int]],
) -> np.ndarray:
    frame = _draw_round_rect_alpha(frame, rect, PANEL_BG_SOFT, alpha=0.82, radius=8)
    _draw_round_rect(frame, rect, color, thickness=2, radius=8)
    text_x = rect[0] + 18
    text_y = rect[1] + 9
    text_items.append((label, (text_x, text_y), TEXT, 22))
    return frame


def _draw_chip(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    text_items: list[tuple[str, tuple[int, int], tuple[int, int, int], int]],
) -> np.ndarray:
    frame = _draw_round_rect_alpha(frame, rect, PANEL_BG_SOFT, alpha=0.9, radius=8)
    _draw_round_rect(frame, rect, ACCENT, thickness=1, radius=8)
    text_items.append((label, (rect[0] + 10, rect[1] + 3), TEXT, 16))
    return frame


def _draw_pill(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    frame = _draw_round_rect_alpha(frame, rect, PANEL_BG_SOFT, alpha=0.86, radius=8)
    _draw_round_rect(frame, rect, color, thickness=2, radius=8)
    return draw_text(frame, label, (rect[0] + 14, rect[1] + 7), TEXT, size=18)


def _draw_progress_bar(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    progress: float,
) -> np.ndarray:
    progress = max(0.0, min(1.0, progress))
    frame = _draw_round_rect_alpha(frame, rect, PANEL_BG_SOFT, alpha=0.85, radius=4)
    left, top, right, bottom = rect
    fill_right = left + int((right - left) * progress)
    if fill_right > left:
        _draw_round_rect(frame, (left, top, fill_right, bottom), SUCCESS, thickness=-1, radius=4)
    return frame


def _draw_glass_panel(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    alpha: float,
) -> np.ndarray:
    frame = _draw_round_rect_alpha(frame, rect, PANEL_BG, alpha=alpha, radius=8)
    _draw_round_rect(frame, rect, (64, 68, 74), thickness=1, radius=8)
    return frame


def _draw_round_rect_alpha(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    alpha: float,
    radius: int,
) -> np.ndarray:
    overlay = frame.copy()
    _draw_round_rect(overlay, rect, color, thickness=-1, radius=radius)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def _draw_round_rect_alpha_in_place(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    alpha: float,
    radius: int,
) -> None:
    overlay = frame.copy()
    _draw_round_rect(overlay, rect, color, thickness=-1, radius=radius)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def _draw_round_rect(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int,
    radius: int,
) -> None:
    left, top, right, bottom = rect
    radius = max(0, min(radius, (right - left) // 2, (bottom - top) // 2))
    if radius == 0:
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
        return

    if thickness < 0:
        cv2.rectangle(frame, (left + radius, top), (right - radius, bottom), color, thickness)
        cv2.rectangle(frame, (left, top + radius), (right, bottom - radius), color, thickness)
        for x, y in (
            (left + radius, top + radius),
            (right - radius, top + radius),
            (left + radius, bottom - radius),
            (right - radius, bottom - radius),
        ):
            cv2.circle(frame, (x, y), radius, color, thickness)
        return

    cv2.line(frame, (left + radius, top), (right - radius, top), color, thickness)
    cv2.line(frame, (left + radius, bottom), (right - radius, bottom), color, thickness)
    cv2.line(frame, (left, top + radius), (left, bottom - radius), color, thickness)
    cv2.line(frame, (right, top + radius), (right, bottom - radius), color, thickness)
    cv2.ellipse(frame, (left + radius, top + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (right - radius, top + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (right - radius, bottom - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(frame, (left + radius, bottom - radius), (radius, radius), 90, 0, 90, color, thickness)


def _status_label(status: str, armed_label: str) -> str:
    if status == "Ready":
        return "Ready"
    if status == "Move" and armed_label:
        return f"Armed: {armed_label}"
    return status


def _status_color(status: str) -> tuple[int, int, int]:
    if status.startswith("Added") or status.startswith("Saved") or status == "Recording":
        return SUCCESS
    if status in {"Try again", "Sentence empty", "Press Record"} or status.startswith("Removed"):
        return WARNING
    if status.startswith("Capturing"):
        return ACCENT
    return MUTED_TEXT


def _point_in_button(x: int, y: int, rect: tuple[int, int, int, int] | None) -> bool:
    if rect is None:
        return False
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


def _should_speak_label(label: str) -> bool:
    normalized = label.strip().casefold()
    return (
        normalized not in BACKGROUND_LABELS
        and normalized not in CLEAR_LABELS
        and normalized not in BACKSPACE_LABELS
        and normalized not in PUNCTUATION_LABELS
    )


def _undo_sentence_word(sentence_words: list[str]) -> str:
    if not sentence_words:
        return ""
    return sentence_words.pop()


def _sentence_text(sentence_words: list[str]) -> str:
    return " ".join(sentence_words).strip()


def _record_sentence(
    sentence_log: str | None,
    sentence_words: list[str],
    event: str,
    source_label: str = "",
    confidence: float | None = None,
) -> SentenceSaveResult:
    return SentenceRecorder(sentence_log).record(sentence_words, event, source_label, confidence)


def _record_and_maybe_speak(
    sentence_recorder: SentenceRecorder,
    sentence_tts: GeorgianTTS,
    sentence_words: list[str],
    event: str,
    source_label: str = "",
    confidence: float | None = None,
) -> tuple[SentenceSaveResult, TTSQueueResult]:
    save_result = sentence_recorder.record(sentence_words, event, source_label, confidence)
    if not save_result.saved or event not in SPOKEN_SAVE_EVENTS:
        return save_result, TTSQueueResult(False, "")
    return save_result, sentence_tts.speak_sentence(_sentence_text(sentence_words), event)


def _with_tts_feedback(feedback: str, tts_result: TTSQueueResult) -> str:
    if not tts_result.message:
        return feedback
    return f"{feedback}; {tts_result.message}"


def _shorten_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[-max_chars:]
    return "..." + text[-(max_chars - 3) :]


def _wrap_text(text: str, max_chars: int, max_lines: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    if not words:
        return [_shorten_text(text, max_chars)]

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == max_lines - 1:
            break

    if current and len(lines) < max_lines:
        remaining = " ".join(words[sum(len(line.split()) for line in lines) :])
        lines.append(_shorten_text(remaining or current, max_chars))

    return lines[:max_lines] or [_shorten_text(text, max_chars)]


def _predict_sequence(
    sequence: list[np.ndarray],
    model: torch.nn.Module,
    labels_by_id: dict[int, str],
    allowed_labels: set[str],
    smoother: PredictionSmoother,
    expected_label: str,
) -> PredictionResult | None:
    device = next(model.parameters()).device
    frames = torch.from_numpy(np.stack(sequence).astype(np.float32)).to(device, non_blocking=True)
    if not _has_dynamic_motion_tensor(frames):
        smoother.reset()
        return None

    x = _add_motion_features_tensor(frames).unsqueeze(0)
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=autocast_enabled):
        probabilities = torch.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id])
    prediction = labels_by_id[class_id]
    if prediction not in allowed_labels:
        smoother.reset()
        return None
    if expected_label and prediction != expected_label:
        smoother.reset()
        return None
    stable_label = smoother.update(prediction, confidence)
    if not stable_label:
        return None
    return PredictionResult(stable_label, confidence)


def _add_motion_features_tensor(frames: torch.Tensor) -> torch.Tensor:
    deltas = torch.zeros_like(frames)
    deltas[1:] = frames[1:] - frames[:-1]
    return torch.cat((frames, deltas), dim=1)


def _has_dynamic_motion_tensor(frames: torch.Tensor) -> bool:
    step_motion = torch.linalg.vector_norm(torch.diff(frames, dim=0), dim=1)
    mean_step_motion = float(step_motion.mean().item())
    start_end_motion = float(torch.linalg.vector_norm(frames[-1] - frames[0]).item())
    active_steps = int(torch.count_nonzero(step_motion >= ACTIVE_STEP_MOTION).item())
    return (
        mean_step_motion >= MIN_MEAN_STEP_MOTION
        and start_end_motion >= MIN_START_END_MOTION
        and active_steps >= MIN_ACTIVE_STEPS
    )


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
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
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
    best_label, best_distance = _best_start_pose_match(vector, start_templates)
    if best_distance <= START_POSE_THRESHOLD:
        return best_label
    return ""


def _best_start_pose_match(
    vector: np.ndarray,
    start_templates: dict[str, np.ndarray],
) -> tuple[str, float]:
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

    return best_label, best_distance
