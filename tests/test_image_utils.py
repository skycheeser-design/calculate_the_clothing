import types
import sys
import os

# ---------------------------------------------------------------------------
# Minimal ``numpy`` replacement with just the features required for these
# tests.  It provides ``zeros`` and ``ones`` constructors that return objects
# exposing a ``shape`` attribute and an ``astype`` method, mimicking the parts of
# ``numpy`` used by the image utilities.
# ---------------------------------------------------------------------------
class _FakeArray:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def astype(self, _):
        return self


np_stub = types.SimpleNamespace(
    zeros=lambda shape, dtype=None: _FakeArray(shape),
    ones=lambda shape, dtype=None: _FakeArray(shape),
    uint8="uint8",
)
sys.modules.setdefault("numpy", np_stub)
import numpy as np

# Ensure repository root is on ``sys.path`` so ``image_utils`` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide a minimal ``PIL`` stub so :mod:`image_utils` can be imported
PIL_stub = types.SimpleNamespace(Image=types.SimpleNamespace(frombytes=lambda *a, **k: None))
sys.modules.setdefault("PIL", PIL_stub)
sys.modules.setdefault("PIL.Image", PIL_stub.Image)


# ---------------------------------------------------------------------------
# Provide a minimal stub of cv2 so that :mod:`image_utils` can be imported
# without the real OpenCV dependency.  Individual tests patch these functions to
# verify behaviour.
# ---------------------------------------------------------------------------
cv2_stub = types.SimpleNamespace(
    MORPH_ELLIPSE=0,
    MORPH_CLOSE=1,
    MORPH_OPEN=2,
    medianBlur=lambda src, ksize: src,
    getStructuringElement=lambda shape, ksize: np.ones(ksize, np.uint8),
    morphologyEx=lambda src, op, kernel: src,
    dilate=lambda src, kernel, iterations=1: src,
)
sys.modules.setdefault("cv2", cv2_stub)

import image_utils


def test_adaptive_kernel_size():
    """Kernel size is chosen based on the shorter image side."""

    assert image_utils._adaptive_kernel_size(np.zeros((400, 800))) == 3
    assert image_utils._adaptive_kernel_size(np.zeros((800, 600))) == 5
    assert image_utils._adaptive_kernel_size(np.zeros((1500, 1400))) == 7


def test_smooth_mask_keep_shape_adaptive_kernel(monkeypatch):
    calls_blur = []
    calls_struct = []

    def fake_blur(src, ksize):
        calls_blur.append(ksize)
        return src

    def fake_struct(shape, ksize):
        calls_struct.append(ksize)
        return np.ones(ksize, np.uint8)

    monkeypatch.setattr(image_utils.cv2, "medianBlur", fake_blur)
    monkeypatch.setattr(image_utils.cv2, "getStructuringElement", fake_struct)
    monkeypatch.setattr(image_utils.cv2, "morphologyEx", lambda s, op, k: s)
    monkeypatch.setattr(image_utils.cv2, "dilate", lambda s, k, i: s)

    mask = np.zeros((400, 800), dtype=np.uint8)  # shorter side < 500 → 3
    image_utils._smooth_mask_keep_shape(mask)

    assert calls_blur == [3]
    assert calls_struct == [(3, 3)]


def test_smooth_mask_keep_shape_defaults(monkeypatch):
    calls_blur = []
    calls_struct = []

    monkeypatch.setattr(image_utils, "_adaptive_kernel_size", lambda m: None)
    monkeypatch.setattr(image_utils.cv2, "medianBlur", lambda s, k: calls_blur.append(k) or s)
    monkeypatch.setattr(
        image_utils.cv2,
        "getStructuringElement",
        lambda shape, ksize: calls_struct.append(ksize) or np.ones(ksize, np.uint8),
    )
    monkeypatch.setattr(image_utils.cv2, "morphologyEx", lambda s, op, k: s)
    monkeypatch.setattr(image_utils.cv2, "dilate", lambda s, k, i: s)

    mask = np.zeros((400, 800), dtype=np.uint8)
    image_utils._smooth_mask_keep_shape(mask)

    assert calls_blur == [5]
    assert calls_struct == [(5, 5), (3, 3)]

