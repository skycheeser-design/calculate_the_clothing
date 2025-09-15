
import cv2
import numpy as np


class NoGarmentDetectedError(RuntimeError):
    """Raised when no suitable garment contour can be found."""


    return contour, measurements

