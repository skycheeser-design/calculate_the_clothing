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


class NoGarmentDetectedError(RuntimeError):
    """Raised when the foreground region resembles paper rather than clothing."""


# ---- 前処理（照明補正・初期マスク・GrabCut） -----------------
def _illum_correction(gray):
    bg = cv2.GaussianBlur(gray, (0, 0), 21)           # 大きめσで背景近似
    corr = cv2.addWeighted(gray, 1.0, bg, -1.0, 128)  # gray - bg + 128
    return np.clip(corr, 0, 255).astype(np.uint8)

def _initial_mask(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lc = _illum_correction(L)

    # Otsu + 適応しきい値（局所）
    _, m1 = cv2.threshold(Lc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m2 = cv2.adaptiveThreshold(
        Lc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 5
    )
    m = cv2.bitwise_or(m1, m2)

    # a,b の K-means（2クラスタ）→ Otsuと重なる方
    ab = np.float32(np.dstack([A, B]).reshape(-1, 2))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1.0)
    _, labels, _ = cv2.kmeans(ab, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(A.shape)
    k0 = (labels == 0).astype(np.uint8) * 255
    k1 = (labels == 1).astype(np.uint8) * 255
    overlap0 = cv2.countNonZero(cv2.bitwise_and(k0, m))
    overlap1 = cv2.countNonZero(cv2.bitwise_and(k1, m))
    k = k0 if overlap0 >= overlap1 else k1

    # 服マスク近傍のエッジ補強
    edges = cv2.Canny(Lc, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)

    init = cv2.bitwise_and(m, k)
    init = cv2.bitwise_or(init, edges)
    init = cv2.morphologyEx(init, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    init = cv2.morphologyEx(init, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return init

def _refine_grabcut(bgr, init_mask):
    # 0:確BG, 2:不確BG, 3:不確FG, 1:確FG
    gc_mask = np.full(init_mask.shape, cv2.GC_PR_BGD, np.uint8)
    gc_mask[init_mask > 0] = cv2.GC_PR_FGD
    core = cv2.erode(init_mask, np.ones((7, 7), np.uint8), 1)
    gc_mask[core > 0] = cv2.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)

    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    # 小穴埋め & 境界維持（粒ノイズ除去は ``_smooth_mask_keep_shape`` へ委ねる）
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # MORPH_OPEN shrinks the garment outline and can eat sleeve edges.
    # Keep boundaries by dilating and delegate noise removal to
    # ``_smooth_mask_keep_shape`` which performs gentle opening.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return mask
# -----------------------------------------------------------------------


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

def _border_touch(bbox, img_shape, margin=3):
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
    """
    前景候補（複数輪郭）から '服らしい' ものを1つ選ぶ。
    条件で除外 → スコアで選択 → 無ければ最大面積にフォールバック。
    """
    H, W = mask_bin.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cnts, hier = cv2.findContours(mask_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cx, cy = W / 2.0, H / 2.0
    keep = []

    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area < 2000:  # 小さすぎは除外（必要に応じて調整）
            continue

        x, y, w, h = cv2.boundingRect(c)
        rect_area = float(w * h) or 1.0

        rectangularity = area / rect_area  # 長方形に近いほど1
        holes = _count_holes(hier, i)
        border = _border_touch((x, y, w, h), (H, W), margin=3)

        roi = gray[y : y + h, x : x + w]
        roi_mask = mask_bin[y : y + h, x : x + w] > 0
        lap_var = _laplacian_var(roi, roi_mask)

        # 紙・板のような領域はスコア計算前に除外
        if _is_paper_like(image_bgr, mask_bin, c):
            continue
        if lap_var < 15.0 and rectangularity > 0.80:
            continue

        # スコア：中央寄り & 面積大 を優遇、長方形/穴/端は減点、質感は加点
        M = cv2.moments(c)
        if M["m00"] > 0:
            mx = M["m10"] / M["m00"]
            my = M["m01"] / M["m00"]
        else:
            mx, my = x + w / 2.0, y + h / 2.0
        center_penalty = np.hypot(mx - cx, my - cy) / max(H, W)  # 0〜1程度

        border_penalty = 0.0
        if border and rectangularity > 0.9:
            border_penalty = 1.6
        elif border:
            border_penalty = 0.8

        score = (
            (area / float(H * W)) * 3.0         # 面積  強
            - rectangularity * 1.5              # 長方形 減点
            - holes * 2.0                       # 穴     減点
            - border_penalty                    # 端接触 減点（長方形はさらに減点）
            - center_penalty * 1.2              # 中央から遠い 減点
            + min(lap_var / 100.0, 2.0) * 0.8   # 質感   加点（上限）
        )
        keep.append((score, i, c))

    if keep:
        keep.sort(key=lambda t: t[0], reverse=True)
        return keep[0][2]
    return max(cnts, key=cv2.contourArea)
# -----------------------------------------------------------------------


def _is_paper_like(image_bgr, mask_bin, contour):
    """Return ``True`` if the contour resembles a plain sheet of paper.

    A region is considered paper-like when it is almost perfectly rectangular,
    lacks visible texture (``lap_var`` < 10) and covers more than half of the
    frame. Such regions are likely background elements (e.g. calibration paper)
    rather than garments.
    """

    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    rect_area = float(w * h) or 1.0
    rectangularity = area / rect_area

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[y : y + h, x : x + w]
    roi_mask = mask_bin[y : y + h, x : x + w] > 0
    lap_var = _laplacian_var(roi, roi_mask)

    H, W = mask_bin.shape[:2]
    coverage = area / float(H * W)

    return rectangularity > 0.95 and lap_var < 10.0 and coverage > 0.5
# -----------------------------------------------------------------------


# ---- 穏やかなスムージング（形状保持。凸包は使わない） ------------
def _smooth_mask_keep_shape(mask):
    m = cv2.medianBlur(mask, 5)
    ell5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ell3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell5)  # 小穴埋め
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  ell3)  # 粒ノイズ除去
    m = cv2.dilate(m, ell3, 1)  # 開処理で削られた境界を少し復元
    return m
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
        prune_threshold = DEFAULT_PRUNE_THRESHOLD

    # --- 1) 初期マスク
    init = _initial_mask(image)
    if debug:
        cv2.imwrite("debug_init_mask.png", init)

    # --- 2) GrabCut & 穏やかなスムージング
    mask = _refine_grabcut(image, init)
    if debug:
        cv2.imwrite("debug_refined_mask.png", mask)
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

    # --- 6) 肩幅：上部25%以内の輪郭から両端点を最遠対で取得 ---
    upper_limit = top_y + int(height * 0.25)
    contour_points = clothes_contour[:, 0, :]
    upper_points = contour_points[contour_points[:, 1] <= y + upper_limit]
    if upper_points.shape[0] < 2:
        raise ValueError("Shoulder line not detected")
    diff = upper_points[:, None, :] - upper_points[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    i, j = np.unravel_index(np.argmax(dist_sq), dist_sq.shape)
    left_shoulder = tuple(upper_points[i])
    right_shoulder = tuple(upper_points[j])
    if left_shoulder[0] > right_shoulder[0]:
        left_shoulder, right_shoulder = right_shoulder, left_shoulder
    shoulder_width = right_shoulder[0] - left_shoulder[0]

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
