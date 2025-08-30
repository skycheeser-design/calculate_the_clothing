import os
import importlib.util
import numpy as np
import cv2

# Load Clothing module from the script without extension
MODULE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Clothing')
spec = importlib.util.spec_from_file_location('clothing', MODULE_PATH)
clothing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clothing)


def create_test_image():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # body
    cv2.rectangle(img, (80, 50), (120, 180), (255, 255, 255), -1)
    # sleeves (triangular) to give unique endpoints
    left_sleeve = np.array([[80, 70], [30, 90], [80, 110]], np.int32)
    right_sleeve = np.array([[120, 70], [170, 90], [120, 110]], np.int32)
    cv2.fillConvexPoly(img, left_sleeve, (255, 255, 255))
    cv2.fillConvexPoly(img, right_sleeve, (255, 255, 255))
    return img


def create_long_sleeve_image():
    img = np.zeros((300, 200, 3), dtype=np.uint8)
    # body with extra length to ensure center column is tallest
    cv2.rectangle(img, (80, 50), (120, 210), (255, 255, 255), -1)
    # sleeves extending downward from under the shoulders
    cv2.rectangle(img, (60, 110), (80, 240), (255, 255, 255), -1)
    cv2.rectangle(img, (120, 110), (140, 240), (255, 255, 255), -1)
    return img


def test_measure_clothes_lengths():
    img = create_test_image()
    contour, measures = clothing.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    # Expected body length between top y=50 and bottom y=180
    assert abs(measures['身丈'] - 130) < 1.0
    # Expected sleeve length from shoulder (80,63) to sleeve end (30,90)
    expected_sleeve = np.hypot(80 - 30, 63 - 90)
    assert abs(measures['袖丈'] - expected_sleeve) < 1.0


def test_measure_clothes_long_sleeve_length():
    img = create_long_sleeve_image()
    contour, measures = clothing.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    # Expected sleeve length from shoulder (80,66) to sleeve end (60,240)
    expected_sleeve = np.hypot(80 - 60, 66 - 240)
    assert abs(measures['袖丈'] - expected_sleeve) < 1.0
