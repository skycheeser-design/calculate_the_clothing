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
        (typically ``k=2``) and each cluster is scored by central occupancy,
        texture and border contact to choose the garment region.
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

    # Score clusters via central occupancy, texture and border contact
    h, w = labels.shape
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    center_y0, center_y1 = int(h * 0.2), int(h * 0.8)
    center_x0, center_x1 = int(w * 0.2), int(w * 0.8)
    center_window = np.zeros_like(labels, dtype=bool)
    center_window[center_y0:center_y1, center_x0:center_x1] = True

    border_mask = np.zeros_like(labels, dtype=bool)
    border_mask[0, :] = border_mask[-1, :] = True
    border_mask[:, 0] = border_mask[:, -1] = True

    occs, lap_vars, borders = [], [], []
    for i in range(k):
        cluster = labels == i
        count = np.count_nonzero(cluster)
        if count == 0:
            occs.append(0.0)
            lap_vars.append(0.0)
            borders.append(1.0)
            continue
        occ = np.count_nonzero(cluster & center_window) / count
        lap_var = float(lap[cluster].var()) if count else 0.0
        border_ratio = np.count_nonzero(cluster & border_mask) / count
        occs.append(occ)
        lap_vars.append(lap_var)
        borders.append(border_ratio)

    lap_max = max(lap_vars) if max(lap_vars) > 0 else 1.0
    scores = []
    for occ, lap_var, border_ratio in zip(occs, lap_vars, borders):
        score = 0.5 * occ + 0.3 * (lap_var / lap_max) - 0.2 * border_ratio
        scores.append(score)

    garment_cluster = int(np.argmax(scores))
    init_mask = np.where(labels == garment_cluster, 1, 0).astype("uint8")

    # 3) Refine with GrabCut
    gc_mask = np.where(init_mask == 1, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype("uint8")
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(smooth, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    # Detect bright low-saturation rectangular regions (e.g. paper backgrounds)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, (0, 0, 200), (180, 25, 255))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    img_area = h * w
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / img_area
        if area_ratio < 0.01 or area_ratio > 0.30:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, rw, rh = cv2.boundingRect(approx)
        aspect = max(rw, rh) / max(min(rw, rh), 1)
        if not (1.21 <= aspect <= 1.61):
            continue
        region_mask = np.zeros_like(thresh)
        cv2.drawContours(region_mask, [approx], -1, 255, -1)
        mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=region_mask)
        if mean_v <= 200 or mean_s >= 25:
            continue
        lap_var = cv2.Laplacian(gray[y : y + rh, x : x + rw], cv2.CV_64F).var()
        if lap_var >= 10:
            continue
        cv2.drawContours(gc_mask, [approx], -1, cv2.GC_BGD, -1)

    # Rerun GrabCut with refined background labels
    cv2.grabCut(smooth, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
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
