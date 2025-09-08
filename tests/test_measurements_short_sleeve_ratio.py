import numpy as np
import cv2

from measurements import measure_clothes


def create_short_sleeve_image():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 50), (120, 180), (255, 255, 255), -1)
    left_sleeve = np.array([[80, 70], [30, 90], [80, 110]], np.int32)
    right_sleeve = np.array([[120, 70], [170, 90], [120, 110]], np.int32)
    cv2.fillConvexPoly(img, left_sleeve, (255, 255, 255))
    cv2.fillConvexPoly(img, right_sleeve, (255, 255, 255))
    return img


def create_long_sleeve_image():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 50), (200, 180), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 70), (100, 150), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 70), (250, 150), (255, 255, 255), -1)
    return img


def test_short_sleeve_excluded_sleeve_length():
    img = create_short_sleeve_image()
    contour, measures = measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    assert "袖丈" not in measures
    assert measures.get("袖タイプ") == "短袖"


def test_long_sleeve_keeps_sleeve_length():
    img = create_long_sleeve_image()
    contour, measures = measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    assert "袖丈" in measures
    assert "袖タイプ" not in measures
