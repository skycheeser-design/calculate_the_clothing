import cv2
import numpy as np


class NoGarmentDetectedError(RuntimeError):
    """Raised when no suitable garment contour can be found."""


def _binary_mask(image: np.ndarray) -> np.ndarray:
    """Return a binary mask of the garment via simple thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Evaluate both normal and inverted thresholds and keep the variant with
    # the larger foreground region.  This handles both light and dark garments.
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts1, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area1 = max((cv2.contourArea(c) for c in cnts1), default=0)
    area2 = max((cv2.contourArea(c) for c in cnts2), default=0)
    return th1 if area1 >= area2 else th2


def _largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def _measure_contour(contour, cm_per_pixel: float) -> dict:
    """Compute simple width/height measurements from ``contour``."""
    x, y, w, h = cv2.boundingRect(contour)
    return {"身幅": w * cm_per_pixel, "身丈": h * cm_per_pixel}


def measure_clothes(image: np.ndarray, cm_per_pixel: float, debug: bool = False):
    """Measure garment dimensions directly from contour information."""
    mask = _binary_mask(image)
    contour = _largest_contour(mask)
    if contour is None:
        raise NoGarmentDetectedError("No garment detected")

    measurements = _measure_contour(contour, cm_per_pixel)

    if debug:
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(dbg, [contour], -1, (0, 255, 0), 2)
        return contour, measurements, dbg

    return contour, measurements

