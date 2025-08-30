"""Background handling utilities such as marker detection and removal."""

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    from rembg import remove
except ImportError:  # pragma: no cover
    remove = None


def detect_marker(image: np.ndarray, marker_size_cm: float = 5.0) -> float:
    """Detect a square calibration marker.

    Returns centimeters per pixel when a marker is found, otherwise ``None``.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cm_per_pixel = None
    best_cnt = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # remove noise
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.9 < aspect_ratio < 1.1 and area > max_area:
                max_area = area
                best_cnt = cnt

    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        cm_per_pixel = marker_size_cm / np.mean([w, h])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Marker", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return cm_per_pixel


def remove_background(image: np.ndarray) -> np.ndarray:
    """Remove background using ``rembg``.

    Raises ``ImportError`` if ``rembg`` is not available.
    """
    if remove is None:
        raise ImportError("rembg is required for background removal")
    result = remove(image)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGR)
