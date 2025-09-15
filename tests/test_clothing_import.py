import importlib.util
import os
from pathlib import Path

import pytest

# Skip this test if required dependencies for importing Clothing are missing
required_modules = ["numpy", "cv2", "PIL", "pillow_heif"]
for mod in required_modules:
    if importlib.util.find_spec(mod) is None:
        pytest.skip(f"{mod} is required for this test", allow_module_level=True)


def test_import_has_no_side_effect(tmp_path):
    output = tmp_path / "clothes_with_measurements.jpg"
    assert not output.exists()

    module_path = Path(__file__).resolve().parent.parent / "Clothing"
    spec = importlib.util.spec_from_file_location("clothing", module_path)
    clothing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clothing)

    assert not output.exists()
