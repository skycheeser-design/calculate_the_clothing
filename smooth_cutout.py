import argparse
import cv2
import numpy as np


def generate_mask(image: np.ndarray) -> np.ndarray:
    """Return a binary mask of the clothing region.

    Parameters
    ----------
    image: np.ndarray
        Input image in BGR colour space.

    Returns
    -------
    np.ndarray
        Binary mask where garment pixels are 255 and background is 0.
    """
    if image is None:
        raise ValueError("image is required")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    return mask


def cutout_clothes(input_path: str, output_path: str) -> np.ndarray:
    """CLI-friendly wrapper that saves the generated mask to ``output_path``."""
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")
    mask = generate_mask(image)
    cv2.imwrite(output_path, mask)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate garment mask and save as PNG")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("output", help="Path to save the mask image")
    args = parser.parse_args()
    cutout_clothes(args.input, args.output)


if __name__ == "__main__":
    main()
