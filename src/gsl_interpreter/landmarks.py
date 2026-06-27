from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from collections.abc import Iterator

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

import cv2
import mediapipe as mp
import numpy as np

from gsl_interpreter import FEATURE_SIZE

_mp_holistic = mp.solutions.holistic
_holistic: object | None = None

LOCAL_RIGHT_SLOT = slice(0, 63)
LOCAL_LEFT_SLOT = slice(63, 126)
BODY_RIGHT_SLOT = slice(126, 189)
BODY_LEFT_SLOT = slice(189, 252)
POSE_SLOT = slice(252, 351)


def extract(frame: np.ndarray) -> np.ndarray | None:
    """Return body-relative landmarks as shape (351,) float32, or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with _suppress_native_stderr():
        results = _get_holistic().process(rgb)

    if results.pose_landmarks is None:
        return None

    right_hand = results.right_hand_landmarks
    left_hand = results.left_hand_landmarks
    if right_hand is None and left_hand is None:
        return None

    pose_points = _landmarks_to_array(results.pose_landmarks)
    origin, scale = _body_frame(pose_points)
    vector = np.zeros(FEATURE_SIZE, dtype=np.float32)

    if right_hand is not None:
        right_points = _landmarks_to_array(right_hand)
        vector[LOCAL_RIGHT_SLOT] = _normalize_hand_shape(right_points, mirror=False)
        vector[BODY_RIGHT_SLOT] = _normalize_to_body(right_points, origin, scale)

    if left_hand is not None:
        left_points = _landmarks_to_array(left_hand)
        vector[LOCAL_LEFT_SLOT] = _normalize_hand_shape(left_points, mirror=True)
        vector[BODY_LEFT_SLOT] = _normalize_to_body(left_points, origin, scale)

    vector[POSE_SLOT] = _normalize_to_body(pose_points, origin, scale)
    return vector


def _landmarks_to_array(landmarks: object) -> np.ndarray:
    return np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark],
        dtype=np.float32,
    )


def _normalize_hand_shape(points: np.ndarray, mirror: bool) -> np.ndarray:
    points = points.copy()
    wrist = points[0].copy()
    points -= wrist

    scale = np.linalg.norm(points[9])
    if scale < 1e-6:
        scale = 1.0
    points /= scale

    if mirror:
        points[:, 0] *= -1.0

    return points.reshape(63).astype(np.float32)


def _normalize_to_body(points: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    normalized = (points - origin) / scale
    return normalized.reshape(-1).astype(np.float32)


def _body_frame(pose_points: np.ndarray) -> tuple[np.ndarray, float]:
    left_shoulder = pose_points[11]
    right_shoulder = pose_points[12]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))

    if shoulder_width < 1e-4:
        left_hip = pose_points[23]
        right_hip = pose_points[24]
        shoulder_width = float(np.linalg.norm(left_hip[:2] - right_hip[:2]))

    return shoulder_center.astype(np.float32), max(shoulder_width, 1e-4)


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


@contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    """Temporarily silence native MediaPipe/TFLite stderr chatter."""
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)
