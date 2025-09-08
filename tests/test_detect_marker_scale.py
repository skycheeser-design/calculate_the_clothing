import importlib.util
from pathlib import Path

import cv2
import numpy as np


def _load_module():
    module_path = Path(__file__).resolve().parent.parent / "Clothing"
    spec = importlib.util.spec_from_file_location("clothing", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_image():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (75, 75), (125, 125), (0, 0, 0), -1)
    return img


def test_detect_marker_returns_scale():
    clothing = _load_module()
    img = _create_image()
    cm_per_pixel = clothing.detect_marker(img.copy(), marker_size_cm=5.0)
    assert cm_per_pixel is not None
    assert abs(cm_per_pixel - 0.1) < 0.02


def test_detect_marker_debug_returns_scale():
    clothing = _load_module()
    img = _create_image()
    cm_per_pixel, debug = clothing.detect_marker(
        img.copy(), marker_size_cm=5.0, debug=True
    )
    assert cm_per_pixel is not None
    assert debug.shape == img.shape
    assert np.any(debug != img)
