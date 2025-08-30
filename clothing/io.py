"""Image I/O utilities with optional HEIC support."""

import os

import cv2
import numpy as np
from PIL import Image

try:  # pragma: no cover - optional dependency
    import pillow_heif
except ImportError:  # pragma: no cover
    pillow_heif = None


def load_image(path: str) -> np.ndarray:
    """Load an image from ``path``.

    Supports regular formats via OpenCV and HEIC images via ``pillow_heif``.
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".heic":
        if pillow_heif is None:
            raise ImportError("pillow_heif is required to load HEIC images")
        heif_file = pillow_heif.read_heif(path)
        img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return cv2.imread(path)
