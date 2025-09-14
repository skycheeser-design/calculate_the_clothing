import argparse
import cv2
import numpy as np


def generate_mask(image: np.ndarray) -> np.ndarray:
    """Return a binary mask of the clothing region.

    The routine performs a colour based segmentation followed by GrabCut and
    morphological cleanup to obtain a stable garment mask.  The processing
    pipeline is intentionally heavier than a naive threshold in order to reduce
    sensitivity to background colours and shadows while preserving important
    concavities of the garment outline.

    Steps
    -----
    1.  Bilateral filtering for edge‑preserving denoising.
    2.  ``a`` and ``b`` channels in Lab space are clustered via K-means
        (``k=2``) to obtain an initial foreground estimate.
    3.  The estimate is refined with :func:`cv2.grabCut`.
    4.  The result is morphologically closed and opened using an elliptical
        kernel.
    5.  A distance transform is normalised and re-thresholded to obtain the
        final binary mask.

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

    # 1) Edge preserving smoothing
    smooth = cv2.bilateralFilter(image, 9, 75, 75)

    # 2) Cluster a,b channels in Lab space
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    ab = lab[:, :, 1:3].reshape((-1, 2)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    _compactness, labels, _centers = cv2.kmeans(
        ab, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(lab.shape[:2])
    h, w = labels.shape
    border = np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]]
    )
    counts = [np.count_nonzero(border == i) for i in range(k)]
    garment_cluster = int(np.argmin(counts))
    init_mask = np.where(labels == garment_cluster, 1, 0).astype("uint8")

    # 3) Refine with GrabCut
    gc_mask = np.where(init_mask == 1, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype("uint8")
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(smooth, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype("uint8")

    # 4) Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 5) Distance transform normalisation & contour based re-binarisation
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    _, dist_bin = cv2.threshold(dist, 0.1, 1.0, cv2.THRESH_BINARY)
    dist_bin = (dist_bin * 255).astype("uint8")
    cnts, _ = cv2.findContours(dist_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(final_mask, [c], -1, 255, -1)
    return final_mask


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
