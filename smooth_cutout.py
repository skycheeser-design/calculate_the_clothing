import cv2
import numpy as np


def generate_mask(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """Generate a binary garment mask using basic image processing.

    The previous implementation relied on a heavy GrabCut based pipeline which
    proved brittle in real world scenarios.  This simplified version merely
    converts the image to greyscale, applies Otsu's threshold and returns the
    largest contour as a filled mask.  It deliberately avoids any learning
    based or iterative refinement steps so the behaviour is deterministic and
    easy to debug.

    Parameters
    ----------
    image: np.ndarray
        Input image in BGR colour space.
    debug: bool, default False
        When ``True`` an intermediate visualisation is written to
        ``debug_mask.png`` for inspection.

    Returns
    -------
    np.ndarray
        Binary mask where garment pixels are ``255`` and background is ``0``.
    """

    if image is None:
        raise ValueError("image is required")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Try both normal and inverted thresholds and keep the variant with the
    # larger foreground region.  This provides a modicum of robustness for
    # both light-on-dark and dark-on-light garments.
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts1, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area1 = max((cv2.contourArea(c) for c in cnts1), default=0)
    area2 = max((cv2.contourArea(c) for c in cnts2), default=0)
    mask = th1 if area1 >= area2 else th2

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [c], -1, 255, -1)

    if debug:
        cv2.imwrite("debug_mask.png", out)

    return out


def cutout_clothes(
    input_path: str,
    mask_output: str,
    masked_output: str | None = None,
    contour_output: str | None = None,
) -> np.ndarray:
    """Generate and save garment mask and optional visualisations.

    Parameters
    ----------
    input_path:
        Path to the source image.
    mask_output:
        File path where the binary mask will be written.
    masked_output:
        Optional path to save the original image with the mask applied.
        The background is filled with black.
    contour_output:
        Optional path to save a visualisation with the detected garment
        contour drawn on top of the original image.

    Returns
    -------
    np.ndarray
        The generated binary mask.
    """

    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")

    mask = generate_mask(image)
    cv2.imwrite(mask_output, mask)

    if masked_output is not None:
        masked = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite(masked_output, masked)

    if contour_output is not None:
        contour_img = image.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(contour_output, contour_img)

    return mask


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate garment mask and optional visualisations"
    )
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("output", help="Path to save the mask image")
    parser.add_argument(
        "--masked-output",
        help="Optional path to save the source image with the mask applied",
    )
    parser.add_argument(
        "--contour-output",
        help="Optional path to save the image with contours overlaid",
    )
    args = parser.parse_args()
    cutout_clothes(
        args.input,
        args.output,
        masked_output=args.masked_output,
        contour_output=args.contour_output,
    )


if __name__ == "__main__":
    main()

