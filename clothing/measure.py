"""Measurement utilities for extracting garment dimensions."""

from heapq import heappush, heappop
import numpy as np
import cv2
from skimage.morphology import skeletonize


def _nearest_skeleton_point(skeleton: np.ndarray, point):
    """Return the skeleton pixel closest to ``point``."""
    ys, xs = np.nonzero(skeleton)
    if xs.size == 0:
        raise ValueError("Skeleton is empty")
    dists = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
    idx = np.argmin(dists)
    return int(xs[idx]), int(ys[idx])


def _shortest_path_length(skeleton: np.ndarray, start, end) -> float:
    """Compute shortest path length between two points on a skeleton."""
    height, width = skeleton.shape
    visited = np.zeros((height, width), dtype=bool)
    dist = np.full((height, width), np.inf)
    sx, sy = start
    ex, ey = end
    dist[sy, sx] = 0.0
    heap = [(0.0, sx, sy)]
    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]
    while heap:
        d, x, y = heappop(heap)
        if visited[y, x]:
            continue
        if (x, y) == (ex, ey):
            return d
        visited[y, x] = True
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx]:
                step = 1.41421356 if dx != 0 and dy != 0 else 1.0
                nd = d + step
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heappush(heap, (nd, nx, ny))
    return np.inf


def _furthest_point(points: np.ndarray, start):
    """Return the furthest point from ``start`` along a skeleton."""
    if points.size == 0:
        start = (int(start[0]), int(start[1]))
        return start, 0.0

    max_x = int(max(points[:, 0].max(), start[0])) + 1
    max_y = int(max(points[:, 1].max(), start[1])) + 1
    skeleton = np.zeros((max_y, max_x), dtype=bool)
    skeleton[points[:, 1], points[:, 0]] = True

    sx, sy = _nearest_skeleton_point(skeleton, (int(start[0]), int(start[1])))

    furthest = (sx, sy)
    max_dist = 0.0
    for x, y in points:
        length = _shortest_path_length(skeleton, (sx, sy), (int(x), int(y)))
        if length > max_dist:
            max_dist = length
            furthest = (int(x), int(y))
    return furthest, float(max_dist)


def measure_clothes(image: np.ndarray, cm_per_pixel: float):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {}
    clothes_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(clothes_contour)
    x, y, w, h = cv2.boundingRect(hull)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [clothes_contour], -1, 255, thickness=-1)
    projection = mask.sum(axis=0)
    center_x = int(np.argmax(projection))
    column_pixels = np.where(mask[:, center_x] > 0)[0]
    if column_pixels.size == 0:
        raise ValueError("Center line not found")
    top_y = int(column_pixels.min())
    bottom_y = int(column_pixels.max())
    height = bottom_y - top_y

    shoulder_y = top_y + int(height * 0.1)
    shoulder_line = mask[shoulder_y:shoulder_y + 5, x:x + w]
    shoulder_points = cv2.findNonZero(shoulder_line)
    if shoulder_points is None:
        raise ValueError("Shoulder line not detected")
    shoulder_xs = shoulder_points[:, 0, 0]
    shoulder_ys = shoulder_points[:, 0, 1]
    left_idx = np.argmin(shoulder_xs)
    right_idx = np.argmax(shoulder_xs)
    left_shoulder = (x + shoulder_xs[left_idx], shoulder_y + shoulder_ys[left_idx])
    right_shoulder = (x + shoulder_xs[right_idx], shoulder_y + shoulder_ys[right_idx])
    shoulder_width = right_shoulder[0] - left_shoulder[0]

    kernel_size = max(3, height // 10)
    vertical_kernel = np.ones((kernel_size, 1), np.uint8)
    torso_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
    torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, vertical_kernel)

    max_width = 0
    left_chest = right_chest = None
    center_rel = center_x - x
    start_y = int(top_y + height * 0.25)
    end_y = int(top_y + height * 0.5)
    for y_pos in range(start_y, end_y):
        row = torso_mask[y_pos, x:x + w]
        xs = np.where(row > 0)[0]
        if xs.size == 0:
            continue
        segments = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
        for seg in segments:
            if seg[0] <= center_rel <= seg[-1]:
                width = seg[-1] - seg[0]
                if width > max_width:
                    max_width = width
                    left_chest = x + seg[0]
                    right_chest = x + seg[-1]
                break
    if max_width == 0:
        raise ValueError("Chest line not detected")
    chest_width = right_chest - left_chest

    skeleton = skeletonize(mask > 0)
    points = np.column_stack(np.nonzero(skeleton)[::-1])
    left_points = points[points[:, 0] < center_x]
    right_points = points[points[:, 0] >= center_x]

    body_length = _shortest_path_length(skeleton, (center_x, top_y), (center_x, bottom_y))

    left_sleeve_end, left_sleeve_length = _furthest_point(left_points, np.array(left_shoulder, dtype=np.float64))
    right_sleeve_end, right_sleeve_length = _furthest_point(right_points, np.array(right_shoulder, dtype=np.float64))

    sleeve_length = (left_sleeve_length + right_sleeve_length) / 2
    measures = {
        "肩幅": shoulder_width * cm_per_pixel,
        "身幅": chest_width * cm_per_pixel,
        "身丈": body_length * cm_per_pixel,
        "袖丈": sleeve_length * cm_per_pixel,
    }
    return hull, measures
