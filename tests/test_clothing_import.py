import importlib
import importlib.util
import os

import pytest

# Skip if optional dependencies are missing
required_modules = ["numpy", "cv2", "PIL", "pillow_heif", "rembg"]
for mod in required_modules:
    if importlib.util.find_spec(mod) is None:
        pytest.skip(f"{mod} is required for this test", allow_module_level=True)


def test_import_has_no_side_effect(tmp_path):
    output = tmp_path / "clothes_with_measurements.jpg"
    assert not output.exists()

    import clothing.io  # noqa: F401
    import clothing.background  # noqa: F401
    import clothing.measure  # noqa: F401
    import clothing.viz  # noqa: F401

    assert not output.exists()
