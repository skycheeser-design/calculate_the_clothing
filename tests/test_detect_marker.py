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


@pytest.mark.parametrize("bg_color, angle", [(255, 0), (120, 0), (255, 45), (120, 30)])
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

