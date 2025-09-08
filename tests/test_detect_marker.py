import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


def _load_module():
    """Load the ``Clothing`` module regardless of its file name.

    The repository stores the main implementation in a file named ``Clothing``
    without the usual ``.py`` extension.  ``importlib`` is therefore used to
    load it in the tests.
    """

    module_path = Path(__file__).resolve().parent.parent / "Clothing"
    spec = importlib.util.spec_from_file_location("clothing", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bg_color, angle", [(255, 0), (120, 0), (0, 0), (255, 45), (120, 30)])
def test_detect_marker_various_conditions(bg_color, angle):
    clothing = _load_module()

    # Create a background with a mild gradient to emulate uneven lighting
    grad = np.tile(
        np.linspace(max(0, bg_color - 20), min(255, bg_color + 20), 200, dtype=np.uint8),
        (200, 1),
    )
    img = cv2.merge([grad, grad, grad])

    # Insert a black square marker at the centre
    cv2.rectangle(img, (75, 75), (125, 125), (0, 0, 0), -1)

    # Add random noise so morphological opening/closing are exercised
    rng = np.random.default_rng(0)
    coords = rng.integers(0, 200, (300, 2))
    for x, y in coords:
        img[y, x] = (255, 255, 255) if rng.random() > 0.5 else (0, 0, 0)

    # Rotate the image if required to check handling of rotated markers
    if angle:
        M = cv2.getRotationMatrix2D((100, 100), angle, 1.0)
        img = cv2.warpAffine(
            img, M, (200, 200), flags=cv2.INTER_LINEAR, borderValue=(bg_color, bg_color, bg_color)
        )

    cm_per_pixel = clothing.detect_marker(img.copy(), marker_size_cm=5.0)

    assert cm_per_pixel is not None
    # The marker is 50x50 pixels → 5cm / 50px = 0.1 cm per pixel.
    assert abs(cm_per_pixel - 0.1) < 0.02


def test_detect_marker_debug_image():
    clothing = _load_module()

    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (75, 75), (125, 125), (0, 0, 0), -1)

    cm_per_pixel, debug_img = clothing.detect_marker(img.copy(), marker_size_cm=5.0, debug=True)

    assert cm_per_pixel is not None
    # The debug image should have the same shape but contain drawings compared to
    # the original.  A simple inequality check is sufficient here because the
    # bounding box and label introduce colour differences.
    assert debug_img.shape == img.shape
    assert np.any(debug_img != img)


@pytest.mark.parametrize("mode", ["perspective", "partial"])
def test_detect_marker_perspective_and_partial(mode):
    clothing = _load_module()

    base = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(base, (75, 75), (125, 125), (0, 0, 0), -1)

    if mode == "perspective":
        pts1 = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])
        pts2 = np.float32([[0, 0], [200, 0], [180, 200], [20, 200]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(base, M, (200, 200))
        marker_pts = np.float32(
            [[75, 75], [125, 75], [125, 125], [75, 125]]
        ).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(marker_pts, M).reshape(-1, 2)
        d12 = np.linalg.norm(warped_pts[0] - warped_pts[1])
        d34 = np.linalg.norm(warped_pts[2] - warped_pts[3])
        d23 = np.linalg.norm(warped_pts[1] - warped_pts[2])
        d41 = np.linalg.norm(warped_pts[3] - warped_pts[0])
        mean_dim = (d12 + d34 + d23 + d41) / 4
        expected_cpp = 5.0 / mean_dim
    else:
        img = base
        cv2.rectangle(img, (75, 75), (125, 80), (255, 255, 255), -1)
        mean_dim = (50 + 45) / 2
        expected_cpp = 5.0 / mean_dim

    cm_per_pixel = clothing.detect_marker(img.copy(), marker_size_cm=5.0)

    assert cm_per_pixel is not None
    assert abs(cm_per_pixel - expected_cpp) < 0.03


def test_detect_marker_ignores_large_dark_regions():
    """Large dark areas resembling sleeves should not be misidentified as markers."""
    clothing = _load_module()

    img = np.full((200, 200, 3), 255, dtype=np.uint8)

    # Simulate a large dark sleeve occupying much of the frame
    cv2.rectangle(img, (0, 0), (150, 150), (0, 0, 0), -1)

    # Place the actual 40x40 marker in the opposite corner
    cv2.rectangle(img, (150, 150), (190, 190), (0, 0, 0), -1)

    cm_per_pixel = clothing.detect_marker(img.copy(), marker_size_cm=5.0)

    # 5 cm marker represented by 40 pixels → 0.125 cm per pixel expected
    assert cm_per_pixel is not None
    assert abs(cm_per_pixel - 0.125) < 0.02


def test_detect_marker_rotated_high_angle():
    """The scale calculation should remain correct for strongly rotated markers."""
    clothing = _load_module()

    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (75, 75), (125, 125), (0, 0, 0), -1)

    M = cv2.getRotationMatrix2D((100, 100), 60, 1.0)
    img = cv2.warpAffine(
        img, M, (200, 200), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )

    cm_per_pixel = clothing.detect_marker(img.copy(), marker_size_cm=5.0)

    assert cm_per_pixel is not None
    assert abs(cm_per_pixel - 0.1) < 0.02


def test_detect_marker_skewed_perspective():
    """Markers viewed under perspective skew should still yield correct scale."""
    clothing = _load_module()

    base = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(base, (75, 75), (125, 125), (0, 0, 0), -1)

    pts1 = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])
    pts2 = np.float32([[0, 0], [200, 0], [170, 200], [30, 200]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(base, M, (200, 200))

    marker_pts = np.float32(
        [[75, 75], [125, 75], [125, 125], [75, 125]]
    ).reshape(-1, 1, 2)
    warped_pts = cv2.perspectiveTransform(marker_pts, M).reshape(-1, 2)
    side_lengths = [
        np.linalg.norm(warped_pts[i] - warped_pts[(i + 1) % 4]) for i in range(4)
    ]
    expected_cpp = 5.0 / np.mean(side_lengths)

    cm_per_pixel = clothing.detect_marker(img.copy(), marker_size_cm=5.0)

    assert cm_per_pixel is not None
    assert abs(cm_per_pixel - expected_cpp) < 0.03

