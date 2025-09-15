"""Garment segmentation and measurement utilities.

This module exposes a small API used throughout the project and by the unit
tests.  The implementation is based on simple image processing primitives so
that it works for a variety of garment colours without relying on heavy
learning based approaches.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from image_utils import _smooth_mask_keep_shape


class NoGarmentDetectedError(RuntimeError):
    """Raised when no suitable garment contour can be found."""


def segment_garment(img: np.ndarray, thresh_debug_path: Optional[str] = None) -> np.ndarray:
    """Return a binary mask separating garment from the background.

    Parameters
    ----------
    img:
        Input image in BGR colour space.
    thresh_debug_path: Optional[str]
        When provided, debug variants of the threshold candidates are written
        next to this base path.  The function will create three images with
        ``*_hsv``, ``*_otsu`` and ``*_otsu_inv`` suffixes representing the
        saturation/darkness mask and the normal/inverted Otsu results
        respectively.  This helps diagnosing segmentation issues by exposing
        all threshold candidates.
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Candidate 1: saturation or dark value
    M1 = np.where((S > 28) | (V < 90), 255, 0).astype(np.uint8)

    # Candidate 2/3: Otsu thresholding (normal and inverted)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    M2 = otsu
    M3 = cv2.bitwise_not(otsu)

    if thresh_debug_path is not None:
        base, ext = os.path.splitext(thresh_debug_path)
        if not ext:
            ext = ".png"
        cv2.imwrite(f"{base}_hsv{ext}", M1)
        cv2.imwrite(f"{base}_otsu{ext}", M2)
        cv2.imwrite(f"{base}_otsu_inv{ext}", M3)

    # Detect large bright paper-like regions to exclude
    paper = np.where((V > 200) & (S < 25), 255, 0).astype(np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    paper_cnts, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_region = np.zeros_like(paper)
    if paper_cnts:
        cnt = max(paper_cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area > 0.2 * h * w:
            cv2.drawContours(paper_region, [cnt], -1, 255, -1)

    def evaluate(mask: np.ndarray):
        mask = mask.copy()
        mask[paper_region == 255] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        coverage = area / (h * w)
        if coverage < 0.05 or coverage > 0.60:
            return None
        pts = cnt.reshape(-1, 2)
        on_border = np.sum(
            (pts[:, 0] == 0)
            | (pts[:, 0] == w - 1)
            | (pts[:, 1] == 0)
            | (pts[:, 1] == h - 1)
        )
        contact = on_border / len(pts)
        if contact > 0.25:
            return None
        chosen = np.zeros((h, w), np.uint8)
        cv2.drawContours(chosen, [cnt], -1, 255, -1)
        return area, chosen

    candidates = []
    for M in (M1, M2, M3):
        res = evaluate(M)
        if res is not None:
            candidates.append(res)

    if not candidates:
        return np.zeros((h, w), np.uint8)

    candidates.sort(key=lambda x: x[0], reverse=True)
    mask = candidates[0][1]

    # Smooth the mask while keeping shape consistent with other modules
    mask = _smooth_mask_keep_shape(mask)
    return mask


def measure_garment(mask: np.ndarray) -> Dict[str, float]:
    """Return garment measurements given a binary ``mask``."""

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise NoGarmentDetectedError("no garment pixels found")

    top, bottom = ys.min(), ys.max()
    height = bottom - top

    # Shoulder width
    shoulder_rows = []
    s_start = int(top + 0.08 * height)
    s_end = int(top + 0.35 * height)
    for y in range(s_start, s_end + 1):
        row = np.where(mask[y] > 0)[0]
        if row.size == 0:
            continue
        width = row[-1] - row[0]
        shoulder_rows.append((width, y, row[0], row[-1]))
    if not shoulder_rows:
        raise NoGarmentDetectedError("failed to locate shoulder line")
    shoulder_rows.sort(key=lambda x: x[0], reverse=True)
    top25 = shoulder_rows[: max(1, len(shoulder_rows) // 4)]
    shoulder_width, sy, sl, sr = top25[len(top25) // 2]
    left_shoulder = (int(sl), int(sy))
    right_shoulder = (int(sr), int(sy))

    # Body width
    b_start = int(top + 0.40 * height)
    b_end = int(top + 0.60 * height)
    widths = []
    for y in range(b_start, b_end + 1):
        row = np.where(mask[y] > 0)[0]
        if row.size:
            widths.append(row[-1] - row[0])
    body_width = float(np.median(widths)) if widths else 0.0

    # Body length
    body_length = float(height)

    # Sleeve length
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2)
    upper_limit = sy + int(0.5 * height)
    ls_pt = np.array(left_shoulder)
    rs_pt = np.array(right_shoulder)
    left_pts = pts[(pts[:, 0] <= ls_pt[0]) & (pts[:, 1] >= sy) & (pts[:, 1] <= upper_limit)]
    right_pts = pts[(pts[:, 0] >= rs_pt[0]) & (pts[:, 1] >= sy) & (pts[:, 1] <= upper_limit)]
    if left_pts.size == 0:
        left_pts = pts[pts[:, 0] <= ls_pt[0]]
    if right_pts.size == 0:
        right_pts = pts[pts[:, 0] >= rs_pt[0]]
    left_cuff = left_pts[np.argmax(np.linalg.norm(left_pts - ls_pt, axis=1))]
    right_cuff = right_pts[np.argmax(np.linalg.norm(right_pts - rs_pt, axis=1))]
    left_len = float(np.linalg.norm(left_cuff - ls_pt))
    right_len = float(np.linalg.norm(right_cuff - rs_pt))
    sleeve_length = (left_len + right_len) / 2.0

    return {
        "shoulder_width": float(shoulder_width),
        "body_width": body_width,
        "body_length": body_length,
        "sleeve_length": sleeve_length,
        "shoulder_line": (left_shoulder, right_shoulder),
        "left_cuff": tuple(int(v) for v in left_cuff),
        "right_cuff": tuple(int(v) for v in right_cuff),
    }


def visualize(img: np.ndarray, mask: np.ndarray, meas: Dict[str, float]) -> np.ndarray:
    """Draw contour and measurement lines on ``img``."""

    vis = img.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (255, 0, 0), 2)

    ls, rs = meas.get("shoulder_line", ((0, 0), (0, 0)))
    lc = meas.get("left_cuff", (0, 0))
    rc = meas.get("right_cuff", (0, 0))
    cv2.line(vis, ls, rs, (0, 255, 0), 2)
    cv2.line(vis, ls, lc, (0, 255, 255), 2)
    cv2.line(vis, rs, rc, (0, 255, 255), 2)

    texts = [
        f"Shoulder: {meas.get('shoulder_width', 0):.1f}px",
        f"Body width: {meas.get('body_width', 0):.1f}px",
        f"Body length: {meas.get('body_length', 0):.1f}px",
        f"Sleeve: {meas.get('sleeve_length', 0):.1f}px",
    ]
    for i, t in enumerate(texts):
        cv2.putText(
            vis,
            t,
            (10, 30 + 20 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def binary_mask(
    image: np.ndarray,
    debug_path: Optional[str] = None,
    threshold_path: Optional[str] = None,
) -> np.ndarray:
    """Generate garment mask and optionally write it to ``debug_path``.

    Parameters
    ----------
    image:
        Source image in BGR format.
    debug_path:
        If provided the final binary mask is written here.
    threshold_path:
        Optional file path where the raw thresholded image used during
        segmentation is saved.  This allows inspection of the Otsu threshold
        result prior to contour filtering.
    """

    mask = segment_garment(image, thresh_debug_path=threshold_path)
    if debug_path is not None:
        cv2.imwrite(debug_path, mask)
    return mask


def largest_contour(mask: np.ndarray, debug_path: Optional[str] = None):
    """Return largest contour in ``mask`` and optionally save a visualisation."""

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise NoGarmentDetectedError("no contour found")
    cnt = max(cnts, key=cv2.contourArea)
    if debug_path is not None:
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        cv2.imwrite(debug_path, vis)
    return cnt


def measure_clothes(
    image: np.ndarray,
    cm_per_pixel: float = 1.0,
    debug_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """High level convenience wrapper used in the tests."""

    mask_path = contour_path = measure_path = threshold_path = None
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        mask_path = os.path.join(debug_dir, "mask.png")
        contour_path = os.path.join(debug_dir, "contour.png")
        measure_path = os.path.join(debug_dir, "measure.png")
        threshold_path = os.path.join(debug_dir, "threshold.png")

    mask = binary_mask(image, mask_path, threshold_path)
    contour = largest_contour(mask, contour_path)
    meas_px = measure_garment(mask)

    if measure_path is not None:
        vis = visualize(image, mask, meas_px)
        cv2.imwrite(measure_path, vis)

    # Convert to cm using provided scale
    meas_cm = {
        "肩幅": meas_px.get("shoulder_width", 0.0) * cm_per_pixel,
        "身幅": meas_px.get("body_width", 0.0) * cm_per_pixel,
        "身丈": meas_px.get("body_length", 0.0) * cm_per_pixel,
        "袖丈": meas_px.get("sleeve_length", 0.0) * cm_per_pixel,
    }
    return contour, meas_cm


__all__ = [
    "NoGarmentDetectedError",
    "segment_garment",
    "measure_garment",
    "visualize",
    "binary_mask",
    "largest_contour",
    "measure_clothes",
]

