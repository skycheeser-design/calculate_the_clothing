import os
import cv2
import numpy as np

from image_utils import load_image


def generate_mask(
    image: np.ndarray,
    debug: bool = False,
    threshold_output: str | None = None,
) -> np.ndarray:
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
    threshold_output: str | None, default None
        Optional file path where the raw threshold image will be saved.  This
        is the binary image prior to contour extraction.

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
    use_inv = area2 > area1
    mask = th2 if use_inv else th1
    if threshold_output is not None:
        base, ext = os.path.splitext(threshold_output)
        if not ext:
            ext = ".png"
        suffix = "_otsu_inv" if use_inv else "_otsu"
        cv2.imwrite(f"{base}{suffix}{ext}", mask)

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
    threshold_output: str | None = None,
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
    threshold_output:
        Optional path to save the raw threshold image prior to contour
        extraction.

    Returns
    -------
    np.ndarray
        The generated binary mask.
    """

    image = load_image(input_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")

    mask = generate_mask(image, threshold_output=threshold_output)
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
    parser.add_argument(
        "--threshold-output",
        help="Optional path to save the raw threshold image",
    )
    args = parser.parse_args()
    cutout_clothes(
        args.input,
        args.output,
        masked_output=args.masked_output,
        contour_output=args.contour_output,
        threshold_output=args.threshold_output,
    )


if __name__ == "__main__":
    main()

