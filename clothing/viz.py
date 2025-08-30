"""Visualization helpers for annotation of measurements."""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_japanese_font(font_path, font_size):
    """Return a Pillow font object suitable for rendering Japanese text."""
    candidates = []
    if font_path:
        candidates.append(font_path)
    else:
        env_font = os.getenv("JP_FONT_PATH")
        if env_font:
            candidates.append(env_font)
        candidates.extend([
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/Library/Fonts/Kosugi-Regular.ttf",
        ])
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=font_size), font_size
            except OSError:
                continue
    return ImageFont.load_default(), 20


def draw_measurements_on_image(image: np.ndarray, measurements: dict, font_path=None, font_size=150):
    """Draw measurement labels on an image using a Japanese-capable font."""
    font, font_size = _load_japanese_font(font_path, font_size)
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    y_offset = 30
    line_height = font_size + 20
    for key, value in measurements.items():
        text = f"{key}: {value:.1f} cm"
        draw.text((30, y_offset), text, font=font, fill=(0, 255, 0))
        y_offset += line_height
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
