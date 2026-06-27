from __future__ import annotations

import numpy as np

from gsl_interpreter import FEATURE_SIZE, NUM_FRAMES


def add_motion_features(sequence: np.ndarray) -> np.ndarray:
    """Return per-frame landmark positions plus per-frame velocity."""
    frames = np.asarray(sequence, dtype=np.float32)
    if frames.shape != (NUM_FRAMES, FEATURE_SIZE):
        raise ValueError(f"Expected {(NUM_FRAMES, FEATURE_SIZE)}, got {frames.shape}")

    deltas = np.zeros_like(frames)
    deltas[1:] = frames[1:] - frames[:-1]
    return np.concatenate([frames, deltas], axis=1).astype(np.float32)
