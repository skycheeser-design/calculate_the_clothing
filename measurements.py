"""Utility functions to measure garments via simple image processing.

The module provides small, easily testable steps:
1. ``binary_mask`` converts an image into a foreground mask and can save the
   result to disk.
2. ``largest_contour`` extracts the main garment contour from the mask and can
   also save a visualisation.
3. ``measure_contour`` derives width/height measurements from the contour.
4. ``measure_clothes`` ties everything together and optionally writes all
   intermediate images for manual inspection.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import cv2
import numpy as np


class NoGarmentDetectedError(RuntimeError):
    """Raised when no suitable garment contour can be found."""


def binary_mask(image: np.ndarray, save_path: str | None = None) -> np.ndarray:
    """Return a binary mask of the garment via thresholding.

    Parameters
    ----------
    image:
        BGR image containing the garment.
    save_path:
        When provided, the mask is written to ``save_path`` for debugging.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Evaluate both normal and inverted thresholds and keep the variant with
    # the larger foreground region. This handles both light and dark garments.
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts1, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area1 = max((cv2.contourArea(c) for c in cnts1), default=0)
    area2 = max((cv2.contourArea(c) for c in cnts2), default=0)
    mask = th1 if area1 >= area2 else th2

    if save_path:
        cv2.imwrite(save_path, mask)

    return mask


def largest_contour(mask: np.ndarray, save_path: str | None = None):
    """Return the largest contour contained in ``mask``.

    ``save_path`` may be specified to write a visualisation of the contour.
    """

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    contour = max(cnts, key=cv2.contourArea)

    if save_path:
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(dbg, [contour], -1, (0, 255, 0), 2)
        cv2.imwrite(save_path, dbg)

    return contour


def measure_contour(contour, cm_per_pixel: float) -> Dict[str, float]:
    """Compute simple width/height measurements from ``contour``."""

    x, y, w, h = cv2.boundingRect(contour)
    return {"身幅": w * cm_per_pixel, "身丈": h * cm_per_pixel}


def measure_clothes(
    image: np.ndarray,
    cm_per_pixel: float,
    debug_dir: str | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Measure garment dimensions directly from contour information.

    When ``debug_dir`` is supplied, the intermediate mask, contour visualisation
    and final bounding box drawing are written to that directory.
    """

    mask = binary_mask(
        image, os.path.join(debug_dir, "mask.png") if debug_dir else None
    )
    contour = largest_contour(
        mask, os.path.join(debug_dir, "contour.png") if debug_dir else None
    )
    if contour is None:
        raise NoGarmentDetectedError("No garment detected")

    measurements = measure_contour(contour, cm_per_pixel)

    if debug_dir:
        dbg = image.copy()
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, "measure.png"), dbg)

    return contour, measurements

