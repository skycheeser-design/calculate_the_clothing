import os
import numpy as np
import cv2

from clothing import viz as clothing_viz


def test_draw_measurements_with_japanese_font():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    measures = {"肩幅": 50.0}
    font_path = os.getenv("JP_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    out = clothing_viz.draw_measurements_on_image(
        img.copy(), measures, font_path=font_path, font_size=20
    )
    # Ensure drawing modified the image
    assert np.any(out != img)
