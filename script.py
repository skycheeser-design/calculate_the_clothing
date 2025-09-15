"""CLI entry point for garment segmentation and measurement."""

import argparse
import os
import cv2

from image_utils import load_image
from measurements import segment_garment, measure_garment, visualize


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment and measure garment")
    parser.add_argument("image", help="input image path")
    parser.add_argument(
        "--output", default="output.png", help="path to save visualised result"
    )
    parser.add_argument(
        "--threshold-output",
        help="optional base path to save raw threshold candidates",
    )
    parser.add_argument(
        "--debug-dir",
        help="directory to save debug images like thresholds and mask",
    )
    args = parser.parse_args()

    img = load_image(args.image)
    if img is None:
        raise SystemExit("Failed to read input image")

    thresh_path = args.threshold_output
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        if thresh_path is None:
            thresh_path = os.path.join(args.debug_dir, "threshold.png")

    mask = segment_garment(img, thresh_debug_path=thresh_path)
    meas = measure_garment(mask)
    vis = visualize(img, mask, meas)
    cv2.imwrite(args.output, vis)

    if args.debug_dir:
        cv2.imwrite(os.path.join(args.debug_dir, "mask.png"), mask)
        cv2.imwrite(os.path.join(args.debug_dir, "measure.png"), vis)

    for k in ["shoulder_width", "body_width", "body_length", "sleeve_length"]:
        print(f"{k}: {meas.get(k, 0):.1f} px")


if __name__ == "__main__":
    main()

