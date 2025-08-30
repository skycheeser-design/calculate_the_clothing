"""Utility functions for sleeve measurements.

This module isolates the sleeve-related computations so they can be
parallelised or even executed on separate workers in the future.
"""

from concurrent.futures import ThreadPoolExecutor
from heapq import heappush, heappop
from typing import Tuple

import cv2
import numpy as np

# Default minimum branch length to keep when pruning skeletons. Exposed as a
# module level constant so callers can tune it if needed.
DEFAULT_PRUNE_THRESHOLD = 20


def prune_skeleton(skeleton: np.ndarray, min_length: int = DEFAULT_PRUNE_THRESHOLD) -> np.ndarray:
    """Remove short dangling branches from a skeleton image.

    The function iteratively traces every end point in ``skeleton`` and removes
    the branch if its length does not exceed ``min_length`` pixels. This helps
    reduce spurious spurs introduced during skeletonisation so sleeve
    measurements operate on a cleaner topology.
    """

    skel = skeleton.copy().astype(bool)
    h, w = skel.shape
    kernel = np.ones((3, 3), np.uint8)

    def _trace_branch(x: int, y: int):
        """Return coordinates along a branch starting from an endpoint."""

        path = []
        px, py = -1, -1
        cx, cy = x, y
        length = 0
        while True:
            path.append((cx, cy))
            if length >= min_length:
                break
            neighbours = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and skel[ny, nx]:
                        if (nx, ny) != (px, py):
                            neighbours.append((nx, ny))
            if len(neighbours) != 1:
                break
            px, py = cx, cy
            cx, cy = neighbours[0]
            length += 1
        return path

    while True:
        nb = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        nb = nb - skel.astype(np.uint8)
        ys, xs = np.where(skel & (nb == 1))
        removed = False
        for x, y in zip(xs, ys):
            branch = _trace_branch(x, y)
            if len(branch) - 1 <= min_length:
                for bx, by in branch:
                    skel[by, bx] = False
                removed = True
        if not removed:
            break
    return skel


def _nearest_skeleton_point(skeleton: np.ndarray, point: Tuple[int, int]) -> Tuple[int, int]:
    """Return the skeleton pixel closest to ``point``."""

    ys, xs = np.nonzero(skeleton)
    if xs.size == 0:
        raise ValueError("Skeleton is empty")
    dists = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
    idx = np.argmin(dists)
    return int(xs[idx]), int(ys[idx])


def _skeleton_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Return pixels with exactly one 8-connected neighbour."""

    ys, xs = np.nonzero(skeleton)
    height, width = skeleton.shape
    endpoints = []
    for x, y in zip(xs, ys):
        x0, x1 = max(0, x - 1), min(width - 1, x + 1)
        y0, y1 = max(0, y - 1), min(height - 1, y + 1)
        # Count including the pixel itself
        if np.count_nonzero(skeleton[y0 : y1 + 1, x0 : x1 + 1]) == 2:
            endpoints.append((x, y))
    return np.array(endpoints, dtype=int).reshape(-1, 2)


def _shortest_path_length(
    skeleton: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]
) -> float:
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
    return float(np.inf)


def _furthest_point(points: np.ndarray, start: np.ndarray):
    """Return the furthest point from ``start`` along a skeleton."""

    if points.size == 0:
        start = (int(start[0]), int(start[1]))
        return start, 0.0

    max_x = int(max(points[:, 0].max(), start[0])) + 1
    max_y = int(max(points[:, 1].max(), start[1])) + 1
    skeleton = np.zeros((max_y, max_x), dtype=bool)
    skeleton[points[:, 1], points[:, 0]] = True

    sx, sy = _nearest_skeleton_point(skeleton, (int(start[0]), int(start[1])))

    endpoints = _skeleton_endpoints(skeleton)

    furthest = (sx, sy)
    max_dist = 0.0
    found_finite = False
    for x, y in endpoints:
        length = _shortest_path_length(skeleton, (sx, sy), (int(x), int(y)))
        if np.isinf(length):
            continue
        found_finite = True
        if length > max_dist:
            max_dist = length
            furthest = (int(x), int(y))
    if not found_finite:
        return (sx, sy), 0.0
    return furthest, float(max_dist)


def compute_sleeve_length(
    left_points: np.ndarray,
    right_points: np.ndarray,
    left_shoulder: Tuple[int, int],
    right_shoulder: Tuple[int, int],
):
    """Return sleeve endpoints and average length using parallel computation."""

    left_start = np.array(left_shoulder, dtype=np.float64)
    right_start = np.array(right_shoulder, dtype=np.float64)

    with ThreadPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(_furthest_point, left_points, left_start)
        right_future = executor.submit(_furthest_point, right_points, right_start)
        left_end, left_length = left_future.result()
        right_end, right_length = right_future.result()

    sleeve_length = (left_length + right_length) / 2
    return left_end, right_end, sleeve_length


__all__ = [
    "_nearest_skeleton_point",
    "_skeleton_endpoints",
    "_shortest_path_length",
    "_furthest_point",
    "compute_sleeve_length",
    "prune_skeleton",
    "DEFAULT_PRUNE_THRESHOLD",
]

