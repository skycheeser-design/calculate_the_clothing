import cv2
import numpy as np
from skimage.morphology import skeletonize

# ---- 追加：照明補正・初期マスク・GrabCutリファイン -----------------
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

    # a,b の K-means で2クラスタ化→Otsuマスクと重なる方を採用
    ab = np.float32(np.dstack([A, B]).reshape(-1, 2))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(ab, 2, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(A.shape)
    k0 = (labels == 0).astype(np.uint8) * 255
    k1 = (labels == 1).astype(np.uint8) * 255
    overlap0 = cv2.countNonZero(cv2.bitwise_and(k0, m))
    overlap1 = cv2.countNonZero(cv2.bitwise_and(k1, m))
    k = k0 if overlap0 >= overlap1 else k1

    # エッジ補強
    edges = cv2.Canny(Lc, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)

    init = cv2.bitwise_and(m, k)
    init = cv2.bitwise_or(init, edges)
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
    # 小穴埋め & 端スパー除去
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return mask
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


def measure_clothes(image, cm_per_pixel, prune_threshold=None, smooth_factor=1.0):
    """Measure key dimensions of the garment contained in ``image``.

    ``cm_per_pixel`` must be a numeric value specifying the conversion from
    pixels to centimetres. Passing ``None`` or a non-numeric value will raise
    ``ValueError``.

    ``smooth_factor`` adjusts the strength of the morphological operations used
    to remove small concavities. ``1.0`` reproduces the previous behaviour,
    smaller values preserve more detail and ``0`` disables the smoothing steps
    entirely.

    When the skeleton representing the garment's centre line is disconnected
    and no path can be found between top and bottom, the body length falls back
    to the bounding-box height instead of returning infinity.
    """

    from numbers import Real
    if cm_per_pixel is None or not isinstance(cm_per_pixel, Real):
        raise ValueError("cm_per_pixel must be a numeric value")

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
    # --- ここから置き換え（切り抜き強化） ---
    # 1) 初期マスク（照明補正 + Otsu/適応 + abクラスタ + エッジ補強）
    init = _initial_mask(image)

    # 2) GrabCut で境界を微修正
    mask = _refine_grabcut(image, init)

    # 3) 凸包とROIを作りつつ、以降の処理に渡す
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Clothes contour not found")

    clothes_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(clothes_contour)
    x, y, w, h = cv2.boundingRect(hull)

    # ROIへ切り出し（従来通り）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray[y:y + h, x:x + w]
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = clothes_contour - [x, y]
    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=-1)
    # --- 置き換えここまで ---
        # --- マスク健全性チェック（任意） ---
    area = cv2.countNonZero(mask)
    if area < (mask.shape[0] * mask.shape[1]) * 0.02:
        raise ValueError("Mask area too small (possible segmentation failure)")

    # 輪郭がギザギザすぎないか（周長/面積）
    perim = cv2.arcLength(cv2.approxPolyDP(clothes_contour, 2.0, True), True)
    if perim / max(area, 1) > 0.2:
        # しきい値は環境で調整。大きいほどギザギザ→背景ノイズ多め
        pass  # 必要ならフォールバックや再試行ロジックをここに

    # 二値マスクから胴体の重心を求め、胴体中央線を推定
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        raise ValueError("Center line not found")
    center_x = int(moments["m10"] / moments["m00"])
    center_x = max(0, min(w - 1, center_x))

    column_pixels = np.where(mask[:, center_x] > 0)[0]
    if column_pixels.size == 0:
        # 重心列に画素が存在しない場合、近傍の列を探索
        offsets = list(range(1, w))
        for off in offsets:
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
    if smooth_factor > 0:
        kernel_size = max(3, int((height // 10) * smooth_factor))
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, kernel_size)
        )
        torso_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
        torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, vertical_kernel)

        # 袖を胴体から切り離すために横方向に細長いカーネルでエロージョンを実施。
        # ここでは一度細く削って連結成分を抽出した後、必要部分のみを復元する。
        horiz_size = max(3, int((w // 8) * smooth_factor))
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (horiz_size, 3)
        )
        torso_eroded = cv2.erode(torso_mask, horizontal_kernel, iterations=1)

        # 中央列と連結している領域のみを残す。袖が裾付近で胴体と繋がって
        # しまうと身幅が過大に測定されるため、胴体の下部 40% を一時的に
        # 無視して連結成分を抽出する。
        torso_mask_cc = torso_eroded.copy()
        bottom_cut = top_y + int(height * 0.6)
        torso_mask_cc[bottom_cut:, :] = 0
        num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
            torso_mask_cc
        )
        center_rel = center_x
        center_labels = np.unique(labels[:, center_rel])
        torso_only = np.zeros_like(torso_eroded)
        for lbl in center_labels:
            if lbl == 0:
                continue
            torso_only[labels == lbl] = 255
        # 抽出した胴体領域のみを元の太さに戻すためダイレーション
        restore_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(3, horiz_size // 2), 3)
        )
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
    if not np.isfinite(body_length):
        # Center line is disconnected or produced an undefined length; fall back
        # to a simple top-to-bottom measurement derived from the bounding-box
        # height so callers never receive ``inf`` or ``nan``.
        body_length = bottom_y - top_y

    measures = {
        "肩幅": shoulder_width * cm_per_pixel,
        "身幅": chest_width * cm_per_pixel,
        "身丈": body_length * cm_per_pixel,
        "袖丈": sleeve_length * cm_per_pixel,
    }

    return hull, measures


__all__ = ["measure_clothes", "_split_sleeve_points"]
