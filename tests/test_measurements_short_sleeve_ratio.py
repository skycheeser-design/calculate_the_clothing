import numpy as np
import cv2

from measurements import measure_clothes


def test_measurement_ignores_small_noise():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 20), (150, 180), (255, 255, 255), -1)
    # small stray dot that should not influence the measurement
    cv2.circle(img, (10, 10), 5, (255, 255, 255), -1)
    contour, measures = measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    assert abs(measures["身幅"] - 100) < 1.0
    assert abs(measures["身丈"] - 160) < 1.0

