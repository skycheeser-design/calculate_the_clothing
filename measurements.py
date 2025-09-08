import cv2
import numpy as np
from skimage.morphology import skeletonize


SHORT_SLEEVE_RATIO_THRESHOLD = 0.35


def _split_sleeve_points(skeleton, left_shoulder, right_shoulder):
    """Split ``skeleton`` into left/right sleeve points via flood fill."""
    from collections import deque
    # Import here to avoid circular dependency with :mod:`sleeve`.
    from sleeve import _nearest_skeleton_point

    height, width = skeleton.shape
    lx, ly = _nearest_skeleton_point(skeleton, left_shoulder)
    rx, ry = _nearest_skeleton_point(skeleton, right_shoulder)

    labels = np.zeros((height, width), dtype=np.uint8)
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


def measure_clothes(image, cm_per_pixel, prune_threshold=None):
    """Measure key dimensions of the garment contained in ``image``."""

    # Import lazily to avoid circular imports when :mod:`sleeve` needs
    # ``measure_clothes`` from this module.  Older installations of
    # :mod:`sleeve` might not provide ``_shortest_path_length`` which is
    # required for computing body length.  To keep ``measure_clothes`` working
    # on such versions we attempt the import and fall back to a local
    # implementation when it fails.
    from sleeve import compute_sleeve_length, prune_skeleton, DEFAULT_PRUNE_THRESHOLD

    try:  # pragma: no cover - exercised only on outdated ``sleeve`` versions
        from sleeve import _shortest_path_length  # type: ignore
    except ImportError:  # pragma: no cover

        def _shortest_path_length(skeleton, start, end):
            from heapq import heappush, heappop

            height, width = skeleton.shape
            visited = np.zeros((height, width), dtype=bool)
            dist = np.full((height, width), np.inf)
            sx, sy = start
            ex, ey = end
            dist[sy, sx] = 0.0
            heap = [(0.0, sx, sy)]
            neighbours = [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),          (1, 0),
                (-1, 1),  (0, 1), (1, 1),
            ]
            while heap:
                d, x, y = heappop(heap)
                if visited[y, x]:
                    continue
                if (x, y) == (ex, ey):
                    return d
                visited[y, x] = True
                for dx, dy in neighbours:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx]:
                        step = 1.41421356 if dx and dy else 1.0
                        nd = d + step
                        if nd < dist[ny, nx]:
                            dist[ny, nx] = nd
                            heappush(heap, (nd, nx, ny))
            return float(np.inf)

    if prune_threshold is None:
        prune_threshold = DEFAULT_PRUNE_THRESHOLD
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # ノイズ除去のためのクロージング処理
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, {}

    clothes_contour = max(contours, key=cv2.contourArea)

    # 凸包で輪郭を滑らかにする
    hull = cv2.convexHull(clothes_contour)
    x, y, w, h = cv2.boundingRect(hull)

    # ROIに切り出し後のマスクを作成
    gray = gray[y:y + h, x:x + w]
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = clothes_contour - [x, y]
    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=-1)

    # 二値マスクから胴体中央線を推定
    projection = mask.sum(axis=0)
    center_x = int(np.argmax(projection))
    column_pixels = np.where(mask[:, center_x] > 0)[0]
    if column_pixels.size == 0:
        raise ValueError("Center line not found")
    top_y = int(column_pixels.min())
    bottom_y = int(column_pixels.max())

    height = bottom_y - top_y

    # 肩幅（上から10%の位置での幅）
    shoulder_y = top_y + int(height * 0.1)
    shoulder_line = mask[shoulder_y:shoulder_y + 5, :]
    shoulder_points = cv2.findNonZero(shoulder_line)
    if shoulder_points is None:
        raise ValueError("Shoulder line not detected")
    shoulder_xs = shoulder_points[:, 0, 0]
    shoulder_ys = shoulder_points[:, 0, 1]
    left_idx = np.argmin(shoulder_xs)
    right_idx = np.argmax(shoulder_xs)
    left_shoulder = (shoulder_xs[left_idx], shoulder_y + shoulder_ys[left_idx])
    right_shoulder = (shoulder_xs[right_idx], shoulder_y + shoulder_ys[right_idx])
    shoulder_width = right_shoulder[0] - left_shoulder[0]

    # 身幅：胴体の25%〜50%の範囲を探索し、中心線と連結した領域のみを測定
    kernel_size = max(3, height // 10)
    vertical_kernel = np.ones((kernel_size, 1), np.uint8)
    torso_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
    torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, vertical_kernel)

    # 袖を胴体から切り離すために横方向に細長いカーネルでエロージョン→ダイレーション
    horiz_size = max(3, w // 8)
    horizontal_kernel = np.ones((1, horiz_size), np.uint8)
    torso_mask = cv2.erode(torso_mask, horizontal_kernel, iterations=1)
    torso_mask = cv2.dilate(torso_mask, horizontal_kernel, iterations=1)

    # 中央列と連結している領域のみを残す
    num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(torso_mask)
    center_rel = center_x
    center_labels = np.unique(labels[:, center_rel])
    torso_only = np.zeros_like(torso_mask)
    for lbl in center_labels:
        if lbl == 0:
            continue
        torso_only[labels == lbl] = 255
    torso_mask = torso_only

    max_width = 0
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
                width = seg[-1] - seg[0]
                if width > max_width:
                    max_width = width
                break

    if max_width == 0:
        raise ValueError("Chest line not detected")
    chest_width = max_width

    # Erode horizontally around the armpit region to weaken the connection
    # between sleeves and torso before skeletonisation. This helps ensure the
    # sleeves are treated as separate branches when extracting their skeletons.
    armpit_start = top_y + int(height * 0.2)
    armpit_end = top_y + int(height * 0.4)
    kernel_width = max(3, w // 30)
    horiz_kernel = np.ones((1, kernel_width), np.uint8)
    mask[armpit_start:armpit_end] = cv2.erode(
        mask[armpit_start:armpit_end], horiz_kernel
    )

    skeleton = skeletonize(mask > 0)

    skeleton = prune_skeleton(skeleton, prune_threshold)
    left_points, right_points = _split_sleeve_points(
        skeleton, left_shoulder, right_shoulder
    )

    _, _, sleeve_length = compute_sleeve_length(
        left_points, right_points, left_shoulder, right_shoulder
    )

    body_length = _shortest_path_length(
        skeleton, (center_x, top_y), (center_x, bottom_y)
    )

    sleeve_ratio = sleeve_length / body_length if body_length else 0

    measures = {
        "肩幅": shoulder_width * cm_per_pixel,
        "身幅": chest_width * cm_per_pixel,
        "身丈": body_length * cm_per_pixel,
    }
    if sleeve_ratio >= SHORT_SLEEVE_RATIO_THRESHOLD:
        measures["袖丈"] = sleeve_length * cm_per_pixel
    else:
        measures["袖タイプ"] = "短袖"

    return hull, measures


__all__ = ["measure_clothes", "_split_sleeve_points"]

