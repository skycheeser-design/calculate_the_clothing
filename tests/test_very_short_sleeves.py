import os
import importlib.util
import numpy as np
import cv2

# Load Clothing module from the script without extension
MODULE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Clothing')
spec = importlib.util.spec_from_file_location('clothing', MODULE_PATH)
clothing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clothing)

def create_very_short_sleeve_image():
    """Create a test image with extremely short sleeves."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 50), (120, 180), (255, 255, 255), -1)
    cv2.rectangle(img, (70, 60), (80, 70), (255, 255, 255), -1)
    cv2.rectangle(img, (120, 60), (130, 70), (255, 255, 255), -1)
    return img

def test_measure_clothes_very_short_sleeve_length():
    img = create_very_short_sleeve_image()
    contour, measures = clothing.measure_clothes(img, cm_per_pixel=1.0)
    assert contour is not None
    assert abs(measures['袖丈'] - 10) < 1.0
