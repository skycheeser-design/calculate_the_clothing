"""CLI entry point for garment segmentation and measurement."""

import argparse
import cv2

from measurements import segment_garment, measure_garment, visualize


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment and measure garment")
    parser.add_argument("image", help="input image path")
    parser.add_argument(
        "--output", default="output.png", help="path to save visualised result"
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit("Failed to read input image")

    mask = segment_garment(img)
    meas = measure_garment(mask)
    vis = visualize(img, mask, meas)
    cv2.imwrite(args.output, vis)

    for k in ["shoulder_width", "body_width", "body_length", "sleeve_length"]:
        print(f"{k}: {meas.get(k, 0):.1f} px")


if __name__ == "__main__":
    main()

