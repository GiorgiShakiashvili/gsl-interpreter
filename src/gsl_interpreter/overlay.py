from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = (
    Path("C:/Windows/Fonts/sylfaen.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
)


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    size: int = 34,
) -> np.ndarray:
    """Draw Unicode text on an OpenCV BGR frame."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _load_font(size)
    rgb = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()
