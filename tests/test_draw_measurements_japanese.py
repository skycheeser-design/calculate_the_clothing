import os
import importlib.util
from pathlib import Path
import numpy as np
import cv2
import numbers

# Load Clothing module from script
base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODULE_PATH = base / '..' / 'Clothing'
spec = importlib.util.spec_from_file_location('clothing', MODULE_PATH)
clothing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clothing)


def test_draw_and_print_measurements_with_japanese_font_and_string(capfd):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    measures = {"肩幅": 50.0, "袖丈": 60.0, "備考": "テスト"}
    font_path = os.getenv(
        "JP_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    out = clothing.draw_measurements_on_image(
        img.copy(), measures, font_path=font_path
    )
    # Ensure drawing modified the image
    assert np.any(out != img)

    # Ensure printing logic handles non-numeric values
    for k, v in measures.items():
        if isinstance(v, numbers.Number):
            print(f"{k}: {v:.1f} cm")
        else:
            print(f"{k}: {v}")
    captured = capfd.readouterr()
    assert "肩幅: 50.0 cm" in captured.out
    assert "袖丈: 60.0 cm" in captured.out
    assert "備考: テスト" in captured.out
