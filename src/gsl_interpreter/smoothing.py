from __future__ import annotations

from collections import deque


class PredictionSmoother:
    """Simple gate; Claude owns final EMA/N-consecutive-frame review."""

    def __init__(self, threshold: float = 0.85, consecutive: int = 5) -> None:
        self.threshold = threshold
        self.consecutive = consecutive
        self.recent: deque[str] = deque(maxlen=consecutive)

    def update(self, prediction: str, confidence: float) -> str | None:
        if confidence < self.threshold:
            self.recent.clear()
            return None

        self.recent.append(prediction)
        if len(self.recent) < self.consecutive:
            return None
        if len(set(self.recent)) == 1:
            return prediction
        return None

    def reset(self) -> None:
        self.recent.clear()
