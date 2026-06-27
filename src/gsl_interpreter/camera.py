from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


class Camera:
    def __init__(
        self,
        index: int = 0,
        width: int | None = None,
        height: int | None = None,
        buffer_size: int = 1,
    ) -> None:
        self.index = index
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        if buffer_size > 0:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        if width:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
