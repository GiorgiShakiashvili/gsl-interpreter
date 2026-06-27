from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError("Could not read frame from camera")
            yield frame

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
