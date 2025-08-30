import numpy as np
import cv2

from clothing import measure as clothing_measure


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
    contour, measures = clothing_measure.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    assert abs(measures['身丈'] - 130) < 1.0
    expected_sleeve = (90 - 63) + (80 - 30)  # vertical + horizontal along skeleton
    assert abs(measures['袖丈'] - expected_sleeve) < 1.0
    return img


def test_measure_clothes_lengths_long():
    img = create_long_sleeve_image()
    contour, measures = clothing_measure.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    expected_sleeve = 80 - 20  # horizontal length along skeleton
    assert abs(measures['袖丈'] - expected_sleeve) < 1.0


def test_measure_clothes_long_sleeve_length():
    img = create_long_sleeve_image()
    contour, measures = clothing_measure.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    expected_sleeve = np.hypot(80 - 60, 66 - 240)
    assert abs(measures['袖丈'] - expected_sleeve) < 1.0
