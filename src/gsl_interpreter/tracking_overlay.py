from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from gsl_interpreter.landmarks import _suppress_native_stderr

_mp_drawing = mp.solutions.drawing_utils
_mp_holistic = mp.solutions.holistic
_holistic: object | None = None

HAND_STYLE = _mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3)
HAND_CONNECTION_STYLE = _mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2)
POSE_STYLE = _mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=3)
POSE_CONNECTION_STYLE = _mp_drawing.DrawingSpec(color=(255, 220, 0), thickness=2)
FACE_STYLE = _mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
FACE_CONNECTION_STYLE = _mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1)


def draw_tracking(frame: np.ndarray) -> np.ndarray:
    """Draw hand, head, and upper-body tracking points on a BGR frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with _suppress_native_stderr():
        results = _get_holistic().process(rgb)

    output = frame.copy()
    _draw_pose(output, results)
    _draw_hands(output, results)
    _draw_head(output, results)
    return output


def _draw_pose(frame: np.ndarray, results: object) -> None:
    pose_landmarks = getattr(results, "pose_landmarks", None)
    if pose_landmarks is None:
        return
    _mp_drawing.draw_landmarks(
        frame,
        pose_landmarks,
        _mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=POSE_STYLE,
        connection_drawing_spec=POSE_CONNECTION_STYLE,
    )


def _draw_hands(frame: np.ndarray, results: object) -> None:
    for landmarks in (
        getattr(results, "left_hand_landmarks", None),
        getattr(results, "right_hand_landmarks", None),
    ):
        if landmarks is None:
            continue
        _mp_drawing.draw_landmarks(
            frame,
            landmarks,
            _mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_STYLE,
            connection_drawing_spec=HAND_CONNECTION_STYLE,
        )


def _draw_head(frame: np.ndarray, results: object) -> None:
    face_landmarks = getattr(results, "face_landmarks", None)
    if face_landmarks is not None:
        _mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            _mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=FACE_STYLE,
            connection_drawing_spec=FACE_CONNECTION_STYLE,
        )
        return

    pose_landmarks = getattr(results, "pose_landmarks", None)
    if pose_landmarks is None:
        return

    height, width = frame.shape[:2]
    for index in (0, 2, 5, 7, 8):
        landmark = pose_landmarks.landmark[index]
        center = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, center, 4, (255, 255, 255), -1)


def _get_holistic() -> object:
    global _holistic
    if _holistic is None:
        with _suppress_native_stderr():
            _holistic = _mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                refine_face_landmarks=False,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
    return _holistic
