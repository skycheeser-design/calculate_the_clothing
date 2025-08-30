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
    """Create a short-sleeve test image."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 50), (120, 180), (255, 255, 255), -1)
    left_sleeve = np.array([[80, 70], [30, 90], [80, 110]], np.int32)
    right_sleeve = np.array([[120, 70], [170, 90], [120, 110]], np.int32)
    cv2.fillConvexPoly(img, left_sleeve, (255, 255, 255))
    cv2.fillConvexPoly(img, right_sleeve, (255, 255, 255))
    return img


def create_long_sleeve_image():
    """Create a long-sleeve test image."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 50), (120, 180), (255, 255, 255), -1)
    # long, thin sleeves extending horizontally
    cv2.rectangle(img, (20, 60), (80, 66), (255, 255, 255), -1)
    cv2.rectangle(img, (120, 60), (180, 66), (255, 255, 255), -1)
    return img


def test_measure_clothes_lengths_short():
    img = create_test_image()
    contour, measures = clothing.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    assert abs(measures['身丈'] - 130) < 1.0
    expected_sleeve = (90 - 63) + (80 - 30)  # vertical + horizontal along skeleton
    assert abs(measures['袖丈'] - expected_sleeve) < 1.0


def test_measure_clothes_lengths_long():
    img = create_long_sleeve_image()
    contour, measures = clothing.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    expected_sleeve = 80 - 20  # horizontal length along skeleton
    assert abs(measures['袖丈'] - expected_sleeve) < 1.0
