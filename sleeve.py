"""Utility functions for sleeve measurements.

This module isolates the sleeve-related computations so they can be
parallelised or even executed on separate workers in the future.
"""

from concurrent.futures import ThreadPoolExecutor
from heapq import heappush, heappop
from typing import Tuple

import numpy as np


def _nearest_skeleton_point(skeleton: np.ndarray, point: Tuple[int, int]) -> Tuple[int, int]:
    """Return the skeleton pixel closest to ``point``."""

    ys, xs = np.nonzero(skeleton)
    if xs.size == 0:
        raise ValueError("Skeleton is empty")
    dists = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
    idx = np.argmin(dists)
    return int(xs[idx]), int(ys[idx])


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

    furthest = (sx, sy)
    max_dist = 0.0
    for x, y in points:
        length = _shortest_path_length(skeleton, (sx, sy), (int(x), int(y)))
        if length > max_dist:
            max_dist = length
            furthest = (int(x), int(y))
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
    "_shortest_path_length",
    "_furthest_point",
    "compute_sleeve_length",
]

