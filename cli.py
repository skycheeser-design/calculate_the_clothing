"""Command-line interface for the clothing measurement pipeline."""

import argparse
import json
import os
from typing import Optional

import cv2

from clothing import io, background, measure, viz


def run_pipeline(image_path: str, cache_dir: str, skip_background: bool = False,
                 skip_measure: bool = False, skip_viz: bool = False,
                 font_path: Optional[str] = None):
    os.makedirs(cache_dir, exist_ok=True)
    img = io.load_image(image_path)

    cm_per_pixel = background.detect_marker(img.copy())
    if cm_per_pixel is None:
        raise SystemExit("Calibration marker not found")

    bg_path = os.path.join(cache_dir, "no_bg.png")
    if skip_background and os.path.exists(bg_path):
        img_no_bg = cv2.imread(bg_path)
    else:
        img_no_bg = background.remove_background(img)
        cv2.imwrite(bg_path, img_no_bg)

    measure_path = os.path.join(cache_dir, "measurements.json")
    if skip_measure and os.path.exists(measure_path):
        with open(measure_path, "r", encoding="utf-8") as f:
            measurements = json.load(f)
        contour = None
    else:
        contour, measurements = measure.measure_clothes(img_no_bg, cm_per_pixel)
        with open(measure_path, "w", encoding="utf-8") as f:
            json.dump(measurements, f, ensure_ascii=False, indent=2)

    if not skip_viz:
        annotated = viz.draw_measurements_on_image(img.copy(), measurements, font_path=font_path, font_size=200)
        if contour is not None:
            cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 2)
        out_path = os.path.join(cache_dir, "clothes_with_measurements.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"Saved annotated image to {out_path}")

    for k, v in measurements.items():
        print(f"{k}: {v:.1f} cm")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clothing measurement pipeline")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--cache-dir", default="cache", help="Directory for intermediate results")
    parser.add_argument("--skip-background", action="store_true", help="Skip background removal if cached")
    parser.add_argument("--skip-measure", action="store_true", help="Skip measurement if cached")
    parser.add_argument("--skip-viz", action="store_true", help="Skip annotation stage")
    parser.add_argument("--font-path", default=None, help="Path to Japanese-capable font")
    args = parser.parse_args(argv)
    run_pipeline(args.image, args.cache_dir, args.skip_background, args.skip_measure, args.skip_viz, args.font_path)


if __name__ == "__main__":
    main()
