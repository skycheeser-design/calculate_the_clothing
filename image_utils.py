import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image
try:  # Optional dependency for HEIC support
    import pillow_heif
except ImportError:  # pragma: no cover
    pillow_heif = None


def load_image(path):
    """Load an image supporting HEIC via pillow_heif if available."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".heic":
        if pillow_heif is None:
            raise ImportError("pillow_heif is required to load HEIC images")
        heif_file = pillow_heif.read_heif(path)
        img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(path)


def _adaptive_kernel_size(img) -> Optional[int]:
    """Return a kernel size based on the image's shorter side.

    Parameters
    ----------
    img: np.ndarray or tuple
        Image array or a ``(height, width)`` tuple.

    Returns
    -------
    Optional[int]
        ``3``, ``5`` or ``7`` based on the shorter dimension.  ``None`` if the
        size cannot be determined.
    """

    shape = getattr(img, "shape", img)
    if shape is None or len(shape) < 2:
        return None

    try:
        h, w = int(shape[0]), int(shape[1])
    except Exception:
        return None

    short = min(h, w)
    if short < 500:
        return 3
    if short <= 1200:
        return 5
    return 7


def _smooth_mask_keep_shape(mask):
    """Gently smooth ``mask`` while preserving its shape.

    The smoothing strength adapts to the resolution of ``mask`` via
    :func:`_adaptive_kernel_size`, ensuring comparable behaviour across input
    sizes.
    """

    k = _adaptive_kernel_size(mask)
    if k is None:
        m = cv2.medianBlur(mask, 5)
        ell5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ell3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell5)  # Fill small holes
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ell3)  # Remove noise
        m = cv2.dilate(m, ell3, 1)  # Restore boundaries
        return m

    m = cv2.medianBlur(mask, k)
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell)  # Fill small holes
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ell)  # Remove noise
    m = cv2.dilate(m, ell, 1)  # Restore boundaries
    return m
