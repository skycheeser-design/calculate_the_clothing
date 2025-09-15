import cv2
import numpy as np

from measurements import binary_mask, largest_contour, measure_clothes


def test_stepwise_debug_outputs(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)

    mask_file = tmp_path / "step_mask.png"
    contour_file = tmp_path / "step_contour.png"

    mask = binary_mask(img, str(mask_file))
    assert mask_file.exists()

    contour = largest_contour(mask, str(contour_file))
    assert contour is not None
    assert contour_file.exists()

    debug_dir = tmp_path / "final"
    debug_dir.mkdir()
    _, measures = measure_clothes(img, cm_per_pixel=1.0, debug_dir=str(debug_dir))
    assert (debug_dir / "mask.png").exists()
    assert (debug_dir / "contour.png").exists()
    assert (debug_dir / "measure.png").exists()

    assert abs(measures["身幅"] - 60) < 1.0
    assert abs(measures["身丈"] - 60) < 1.0

