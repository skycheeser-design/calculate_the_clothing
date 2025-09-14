# -*- coding: utf-8 -*-
"""
refined_measurements.py
- 服らしさスコアで輪郭選択（紙・板を除外）
- 形状を保つ穏やかなスムージング（凸包は使わない）
- debug=True で中間画像保存
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from smooth_cutout import generate_mask
from image_utils import _smooth_mask_keep_shape



class NoGarmentDetectedError(RuntimeError):
    """Raised when the foreground region resembles paper rather than clothing."""


# ---- Constants controlling garment-likeness heuristics -----------------
MIN_AREA_RATIO = 0.05
MARKER_SAT_THRESHOLD = 60
MARKER_AREA_RATIO = 0.15
RECTANGULARITY_HIGH = 0.90
RECTANGULARITY_LAP = 0.80
LAP_VAR_LOW = 15.0
PAPER_RECT_THRESHOLD = 0.95
PAPER_LAP_VAR_THRESHOLD = 10.0
PAPER_COVERAGE_THRESHOLD = 0.15
BORDER_MARGIN = 3


# ---- “服らしさ”で輪郭選択（紙/板を除外） -----------------------
def _count_holes(hierarchy, idx):
    """hierarchy=RETR_CCOMP の配列から idx の子（穴）数を数える。"""
    if hierarchy is None:
        return 0
    holes = 0
    child = hierarchy[0][idx][2]  # FirstChild
    while child != -1:
        holes += 1
        child = hierarchy[0][child][0]  # Next
    return holes

def _border_touch(bbox, img_shape, margin=BORDER_MARGIN):
    x, y, w, h = bbox
    H, W = img_shape[:2]
    return x <= margin or y <= margin or (x + w) >= W - margin or (y + h) >= H - margin

def _laplacian_var(gray_roi, roi_mask):
    # 質感の強さ（紙は極めて低い）
    if gray_roi.size == 0 or np.count_nonzero(roi_mask) == 0:
        return 0.0
    lap = cv2.Laplacian(gray_roi, cv2.CV_32F, ksize=3)
    return float(np.var(lap[roi_mask]))

def _select_garment_contour(image_bgr, mask_bin):
    """Return the contour most likely to be the garment.

    The routine first evaluates external contours (ignoring holes).  If no
    contour passes the garment-likeness checks, it retries using
    ``RETR_CCOMP`` to incorporate hole information.  The existing filtering
    rules for area, border contact and marker colour are preserved.
    """

    H, W = mask_bin.shape[:2]
    frame_area = H * W

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    def _evaluate(cnts, hier):
        candidates = []
        for i, c in enumerate(cnts):
            area = cv2.contourArea(c)
            if area < frame_area * MIN_AREA_RATIO:
                continue

            x, y, w, h = cv2.boundingRect(c)
            rect_area = float(w * h) or 1.0

            rectangularity = area / rect_area  # 長方形に近いほど1
            holes = _count_holes(hier, i)
            border = _border_touch((x, y, w, h), (H, W), margin=BORDER_MARGIN)

            roi = gray[y : y + h, x : x + w]
            roi_mask = mask_bin[y : y + h, x : x + w] > 0
            lap_var = _laplacian_var(roi, roi_mask)

            roi_hsv = hsv[y : y + h, x : x + w]
            sat_roi = roi_hsv[..., 1]
            sat_mean = sat_roi[roi_mask].mean() if roi_mask.any() else 0.0
            if sat_mean > MARKER_SAT_THRESHOLD and area < frame_area * MARKER_AREA_RATIO:
                continue

            # 固定の除外規則（板/紙を弾く）
            if rectangularity > RECTANGULARITY_HIGH and (border or holes >= 1):
                continue
            if lap_var < LAP_VAR_LOW and rectangularity > RECTANGULARITY_LAP:
                continue

            candidates.append(c)
        if candidates:
            return max(candidates, key=cv2.contourArea)
        return None

    cnts, hier = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = _evaluate(cnts, hier)
    if best is not None:
        return best

    cnts, hier = cv2.findContours(mask_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = _evaluate(cnts, hier)
    if best is not None:
        return best
    return max(cnts, key=cv2.contourArea)
# -----------------------------------------------------------------------


def _is_paper_like(image_bgr, mask_bin, contour):
    """Return ``True`` if the contour resembles a plain sheet of paper.

    A region is considered paper-like when it is almost perfectly rectangular,
    lacks visible texture (``lap_var`` < 10), covers a significant portion of the
    frame and is very bright with little colour. Such regions are likely
    background elements (e.g. calibration paper) rather than garments.
    """

    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    rect_area = float(w * h) or 1.0
    rectangularity = area / rect_area

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[y : y + h, x : x + w]
    roi_mask = mask_bin[y : y + h, x : x + w] > 0
    lap_var = _laplacian_var(roi, roi_mask)

    roi_bgr = image_bgr[y : y + h, x : x + w]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = roi_hsv[..., 1][roi_mask].mean() if roi_mask.any() else 0.0
    val_mean = roi_hsv[..., 2][roi_mask].mean() if roi_mask.any() else 0.0

    H, W = mask_bin.shape[:2]
    coverage = area / float(H * W)

    return (
        rectangularity > PAPER_RECT_THRESHOLD
        and lap_var < PAPER_LAP_VAR_THRESHOLD
        and coverage > PAPER_COVERAGE_THRESHOLD
        and val_mean > 200
        and sat_mean < 25
    )
# -----------------------------------------------------------------------



def _split_sleeve_points(skeleton, left_shoulder, right_shoulder):
    """Split ``skeleton`` into left/right sleeve points via flood fill."""
    from collections import deque
    # Import here to avoid circular dependency with :mod:`sleeve`.
    from sleeve import _nearest_skeleton_point

    height, width = skeleton.shape
    lx, ly = _nearest_skeleton_point(skeleton, left_shoulder)
    rx, ry = _nearest_skeleton_point(skeleton, right_shoulder)

    labels = np.zeros((height, width), dtype=np.uint8)
    from collections import deque
    queue = deque([(lx, ly, 1), (rx, ry, 2)])
    labels[ly, lx] = 1
    labels[ry, rx] = 2

    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),          (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]

    while queue:
        x, y, lbl = queue.popleft()
        if labels[y, x] == 3:
            continue

        conflict = False
        to_add = []
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx]:
                lab = labels[ny, nx]
                if lab not in (0, lbl, 3):
                    labels[ny, nx] = 3
                    labels[y, x] = 3
                    conflict = True
                else:
                    to_add.append((nx, ny, lab))
        if conflict:
            continue
        for nx, ny, lab in to_add:
            if lab == 0:
                labels[ny, nx] = lbl
                queue.append((nx, ny, lbl))

    ly, lx = np.where(labels == 1)
    ry, rx = np.where(labels == 2)
    left_points = np.column_stack((lx, ly))
    right_points = np.column_stack((rx, ry))
    return left_points, right_points


def measure_clothes(
    image,
    cm_per_pixel,
    prune_threshold=None,
    smooth_factor=1.0,
    vertical_kernel_size=None,
    horizontal_kernel_size=None,
    debug=False,
):
    """Measure key dimensions of the garment contained in ``image``.

    Raises
    ------
    NoGarmentDetectedError
        If the segmented foreground resembles a plain sheet of paper rather
        than an actual garment.
    """
    from numbers import Real
    if cm_per_pixel is None or not isinstance(cm_per_pixel, Real):
        raise ValueError("cm_per_pixel must be a numeric value")

    # Lazy import (to avoid circular import issues)
    from sleeve import compute_sleeve_length, prune_skeleton, DEFAULT_PRUNE_THRESHOLD
    try:
        from sleeve import _shortest_path_length  # type: ignore
    except ImportError:
        def _shortest_path_length(skeleton, start, end):
            from heapq import heappush, heappop
            height, width = skeleton.shape
            visited = np.zeros((height, width), dtype=bool)
            dist = np.full((height, width), np.inf)
            sx, sy = start; ex, ey = end
            dist[sy, sx] = 0.0
            heap = [(0.0, sx, sy)]
            neighbours = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
            while heap:
                d, x, y = heappop(heap)
                if visited[y, x]: continue
                if (x, y) == (ex, ey): return d
                visited[y, x] = True
                for dx, dy in neighbours:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx]:
                        step = 1.41421356 if dx and dy else 1.0
                        nd = d + step
                        if nd < dist[ny, nx]:
                            dist[ny, nx] = nd
                            heappush(heap, (nd, nx, ny))
            return float(np.inf)

    if prune_threshold is None:
        prune_threshold = max(DEFAULT_PRUNE_THRESHOLD, image.shape[0] // 80)

    # --- Generate garment mask ---
    mask = generate_mask(image)
    mask = _smooth_mask_keep_shape(mask)
    if debug:
        cv2.imwrite("debug_smooth_mask.png", mask)

    # --- 3) 服らしい輪郭を選ぶ（紙/板を除外）
    clothes_contour = _select_garment_contour(image, mask)
    if clothes_contour is None:
        raise ValueError("Clothes contour not found")
    if _is_paper_like(image, mask, clothes_contour):
        raise NoGarmentDetectedError("No garment detected")

    if debug:
        dbg = image.copy()
        cv2.drawContours(dbg, [clothes_contour], -1, (0, 255, 0), 2)
        cv2.imwrite("debug_contours.png", dbg)

    # --- 4) ROIマスクを作成（凹みは保持） ---
    x, y, w, h = cv2.boundingRect(clothes_contour)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray[y:y + h, x:x + w]
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = clothes_contour - [x, y]
    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=-1)

    # 健全性チェック
    area = cv2.countNonZero(mask)
    if area < (mask.shape[0] * mask.shape[1]) * 0.02:
        raise ValueError("Mask area too small (possible segmentation failure)")

    # --- 5) 中央線と上端・下端を推定 ---
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        raise ValueError("Center line not found")
    center_x = int(moments["m10"] / moments["m00"])
    center_x = max(0, min(w - 1, center_x))

    column_pixels = np.where(mask[:, center_x] > 0)[0]
    if column_pixels.size == 0:
        for off in range(1, w):
            for sign in (-1, 1):
                cx = center_x + off * sign
                if 0 <= cx < w:
                    column_pixels = np.where(mask[:, cx] > 0)[0]
                    if column_pixels.size > 0:
                        center_x = cx
                        break
            if column_pixels.size > 0:
                break
        if column_pixels.size == 0:
            raise ValueError("Center line not found")

    top_y = int(column_pixels.min())
    bottom_y = int(column_pixels.max())
    height = bottom_y - top_y

    # --- 6) 肩幅：上部10%〜35%を走査し、上位30%幅の中央値 ---
    shoulder_rows = []
    start_y = top_y + int(height * 0.10)
    end_y = top_y + int(height * 0.35)
    start_y = max(0, start_y)
    end_y = min(h - 1, end_y)
    for yy in range(start_y, end_y + 1):
        row = mask[yy]
        xs = np.where(row > 0)[0]
        if xs.size >= 2:
            shoulder_rows.append((yy, xs[0], xs[-1], xs[-1] - xs[0]))
    if not shoulder_rows:
        raise ValueError("Shoulder line not detected")
    shoulder_rows.sort(key=lambda r: r[3], reverse=True)
    top_n = max(1, int(np.ceil(len(shoulder_rows) * 0.30)))
    top_rows = shoulder_rows[:top_n]
    widths = [r[3] for r in top_rows]
    shoulder_width = float(np.median(widths))
    yy, left_x, right_x, _ = min(top_rows, key=lambda r: abs(r[3] - shoulder_width))
    left_shoulder = (int(left_x), int(yy))
    right_shoulder = (int(right_x), int(yy))

    # --- 7) 身幅（25%〜50%帯で中央と連結した幅の中央値） ---
    if smooth_factor > 0 or vertical_kernel_size is not None or horizontal_kernel_size is not None:
        kernel_size = (
            vertical_kernel_size
            if vertical_kernel_size is not None
            else max(3, int((height // 10) * smooth_factor))
        )
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, kernel_size))
        torso_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
        torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, vertical_kernel)

        horiz_size = (
            horizontal_kernel_size
            if horizontal_kernel_size is not None
            else max(3, int((w // 8) * smooth_factor))
        )
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (horiz_size, 3))
        torso_eroded = cv2.erode(torso_mask, horizontal_kernel, iterations=1)

        torso_mask_cc = torso_eroded.copy()
        bottom_cut = top_y + int(height * 0.6)
        torso_mask_cc[bottom_cut:, :] = 0
        num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(torso_mask_cc)
        center_rel = center_x
        torso_only = np.zeros_like(torso_eroded)
        for lbl in np.unique(labels[:, center_rel]):
            if lbl != 0:
                torso_only[labels == lbl] = 255
        restore_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, horiz_size // 2), 3))
        torso_mask = cv2.dilate(torso_only, restore_kernel, iterations=1)
    else:
        torso_mask = mask.copy()
        center_rel = center_x

    widths = []
    start_y = int(top_y + height * 0.25)
    end_y = int(top_y + height * 0.5)
    for y_pos in range(start_y, end_y):
        row = torso_mask[y_pos]
        xs = np.where(row > 0)[0]
        if xs.size == 0:
            continue
        segments = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
        for seg in segments:
            if seg[0] <= center_rel <= seg[-1]:
                widths.append(seg[-1] - seg[0])
                break
    if not widths:
        raise ValueError("Chest line not detected")
    chest_width = float(np.median(widths))

    # --- 8) 袖丈・身丈（スケルトン） ---
    armpit_start = top_y + int(height * 0.2)
    armpit_end = top_y + int(height * 0.4)
    kernel_width = max(3, w // 30)
    horiz_kernel = np.ones((1, kernel_width), np.uint8)
    mask[armpit_start:armpit_end] = cv2.erode(mask[armpit_start:armpit_end], horiz_kernel)

    skeleton = skeletonize(mask > 0)
    skeleton = prune_skeleton(skeleton, prune_threshold)

    left_points, right_points = _split_sleeve_points(skeleton, left_shoulder, right_shoulder)
    _, _, sleeve_length = compute_sleeve_length(left_points, right_points, left_shoulder, right_shoulder)

    body_length = _shortest_path_length(skeleton, (center_x, top_y), (center_x, bottom_y))
    if not np.isfinite(body_length):
        body_length = bottom_y - top_y

    measures = {
        "肩幅": shoulder_width * cm_per_pixel,
        "身幅": chest_width * cm_per_pixel,
        "身丈": body_length * cm_per_pixel,
        "袖丈": sleeve_length * cm_per_pixel,
    }

    # 輪郭（凹み保持）と採寸値を返す
    return clothes_contour, measures


__all__ = ["measure_clothes", "_split_sleeve_points", "NoGarmentDetectedError"]
