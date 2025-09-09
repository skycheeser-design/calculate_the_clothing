import os
import importlib.util
import numpy as np
import cv2
import pytest
from sleeve import compute_sleeve_length
from measurements import _split_sleeve_points, NoGarmentDetectedError

# Load Clothing module from the script without extension
MODULE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Clothing')
spec = importlib.util.spec_from_file_location('clothing', MODULE_PATH)
clothing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clothing)


def create_test_image():
    """Create a short-sleeve test image."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 50), (120, 180), (255, 255, 255), -1)
    left_sleeve = np.array([[80, 70], [30, 90], [80, 110]], np.int32)
    right_sleeve = np.array([[120, 70], [170, 90], [120, 110]], np.int32)
    cv2.fillConvexPoly(img, left_sleeve, (255, 255, 255))
    cv2.fillConvexPoly(img, right_sleeve, (255, 255, 255))
    return img


def create_long_sleeve_image():

    img = create_test_image()
    contour, measures = clothing.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    assert abs(measures['身丈'] - 130) < 1.0
    expected_sleeve = (90 - 63) + (80 - 30)  # vertical + horizontal along skeleton
    assert abs(measures['袖丈'] - expected_sleeve) < 1.0
    return img


def create_long_sleeve_width_image():
    """Create an image with long sleeves to test chest width calculation."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 50), (200, 180), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 70), (100, 150), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 70), (250, 150), (255, 255, 255), -1)
    return img


def create_very_long_sleeve_image():
    """Create an image where sleeves extend far beyond the torso."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 230), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 50), (100, 280), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 50), (250, 280), (255, 255, 255), -1)
    return img


def create_chest_width_outlier_image():
    """Create an image with a small protrusion around the chest area."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 50), (120, 180), (255, 255, 255), -1)
    # Add a narrow triangle on the left side that should be ignored
    protrusion = np.array([[80, 100], [30, 110], [80, 120]], np.int32)
    cv2.fillConvexPoly(img, protrusion, (255, 255, 255))
    return img


def test_measure_clothes_lengths_long():
    img = create_long_sleeve_image()
    contour, measures = clothing.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    expected_sleeve = 80 - 20  # horizontal length along skeleton
    assert abs(measures['袖丈'] - expected_sleeve) < 1.0


def test_measure_clothes_long_sleeve_length():
    img = create_long_sleeve_image()
    contour, measures = clothing.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    # Expected sleeve length from shoulder (80,66) to sleeve end (60,240)
    expected_sleeve = np.hypot(80 - 60, 66 - 240)
    assert abs(measures['袖丈'] - expected_sleeve) < 1.0


def test_compute_sleeve_length_disconnected_branch():
    left_points = np.array([[0, y] for y in range(5)], dtype=int)
    right_line = [[50, y] for y in range(5)]
    right_square = [[10, 10], [10, 11], [11, 10], [11, 11]]
    right_points = np.array(right_line + right_square, dtype=int)
    left_shoulder = (0, 0)
    right_shoulder = (10, 10)
    left_end, right_end, sleeve_length = compute_sleeve_length(
        left_points, right_points, left_shoulder, right_shoulder
    )
    assert np.isfinite(sleeve_length)
    assert right_end == right_shoulder
    assert np.isclose(sleeve_length, 2.0)



def test_measure_clothes_disconnected_skeleton_fallback(monkeypatch):
    img = create_test_image()

    # Simulate a disconnected skeleton by forcing the shortest path to return
    # infinity. The measurement should fall back to the bounding-box height
    # instead of propagating ``inf``.
    monkeypatch.setattr(
        "sleeve._shortest_path_length", lambda *args, **kwargs: float("inf")
    )

    contour, measures = clothing.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    assert np.isfinite(measures["身丈"])
    # The bounding-box height of ``create_test_image`` is 130 pixels.
    assert abs(measures["身丈"] - 130) < 1.0


def test_measure_clothes_chest_width_with_long_sleeves():
    img = create_very_long_sleeve_image()
    contour, measures = clothing.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    assert abs(measures["身幅"] - 100) < 1.0


def test_measure_clothes_chest_width_ignores_outlier():
    img = create_chest_width_outlier_image()
    contour, measures = clothing.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    assert abs(measures["身幅"] - 40) < 1.0


def test_measure_clothes_rejects_paper():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (180, 180), (255, 255, 255), -1)
    with pytest.raises(NoGarmentDetectedError):
        clothing.measure_clothes(img, cm_per_pixel=1.0)


