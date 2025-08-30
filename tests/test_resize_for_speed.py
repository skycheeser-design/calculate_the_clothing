import os
import importlib.util
import numpy as np

MODULE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Clothing')
spec = importlib.util.spec_from_file_location('clothing', MODULE_PATH)
clothing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clothing)


def test_resize_for_speed_scales_and_adjusts():
    img = np.zeros((2000, 1000, 3), dtype=np.uint8)
    resized, cpp = clothing.resize_for_speed(img, cm_per_pixel=2.0, max_size=1200)
    assert max(resized.shape[:2]) == 1200
    expected_cpp = 2.0 / (1200 / 2000)
    assert abs(cpp - expected_cpp) < 1e-6
