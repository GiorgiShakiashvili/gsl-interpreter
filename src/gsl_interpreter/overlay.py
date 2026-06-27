from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GEORGIAN_FONT_PATHS = (
    Path("C:/Windows/Fonts/sylfaen.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
)
LATIN_FONT_PATHS = (
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/sylfaen.ttf"),
)
TextItem = tuple[str, tuple[int, int], tuple[int, int, int], int]


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    size: int = 34,
) -> np.ndarray:
    """Draw Unicode text on an OpenCV BGR frame."""
    return draw_text_batch(frame, [(text, position, color, size)])


def draw_text_batch(frame: np.ndarray, items: list[TextItem]) -> np.ndarray:
    """Draw multiple Unicode strings with a single Pillow conversion."""
    if not items:
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for text, position, color, size in items:
        font = _load_font(size, text)
        rgb = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _load_font(size: int, text: str) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_paths = GEORGIAN_FONT_PATHS if _contains_georgian(text) else LATIN_FONT_PATHS
    for path in font_paths:
        if path.exists():
            return _cached_font(str(path), size)
    return ImageFont.load_default()


@lru_cache(maxsize=64)
def _cached_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _contains_georgian(text: str) -> bool:
    return any("\u10a0" <= char <= "\u10ff" for char in text)
