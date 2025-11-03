# IKT213 – Assignment 4 (Local features + Harris)
# Outputs written next to this file:
#   01_harris.png
#   02_aligned.png
#   03_matches.png   (and 03_matches.jpg as a fallback)

import os
import cv2 as cv
import numpy as np


# ----------------------------- Utils -----------------------------
def _here(*parts) -> str:
    """Path relative to this script folder."""
    return os.path.join(os.path.dirname(__file__), *parts)


def _save_png(name: str, img: np.ndarray):
    path = _here(name)
    ok = cv.imwrite(path, img)
    if not ok:
        raise IOError(f"Failed to write: {path}")
    print(f"Saved {name:>18}  shape={img.shape}  ->  {path}")


def _maybe_downscale(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """If image is very large, scale it down to speed up & stabilize matching."""
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / float(s)
    resized = cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
    return resized


# ----------------------- 1) Harris corners -----------------------
def harris_corners(reference_bgr: np.ndarray) -> np.ndarray:
    """
    Harris corner detection:
      gray -> blur -> cornerHarris -> dilate -> threshold -> mark red
    Returns BGR with corners painted red.
    """
    out = reference_bgr.copy()
    gray = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    gray_f32 = np.float32(gray)

    dst = cv.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.04)
    dst = cv.dilate(dst, None)

    out[dst > 0.01 * dst.max()] = (0, 0, 255)  # red
    return out


# --------------- 2) Feature-based alignment (robust) --------------
def align_feature_based(
    image_to_align_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    method: str = "ORB",
    max_features: int = 1500,
    good_match_percent: float = 0.15,
):
    """
    Align `image_to_align_bgr` onto `reference_bgr` using features + homography.
    - Uses KNN + Lowe ratio filtering
    - Limits drawn matches to avoid huge output images
    Parameters per assignment:
      ORB : max_features=1500, good_match_percent=0.15
      SIFT: max_features=10,   good_match_percent=0.70
    Returns: (aligned_bgr, matches_vis)
    """
    # Optionally downscale to make matching more stable on very large scans
    img1 = _maybe_downscale(image_to_align_bgr)
    img2 = _maybe_downscale(reference_bgr)

    method = method.upper()
    if method == "ORB":
        detector = cv.ORB_create(nfeatures=max_features)
        norm = cv.NORM_HAMMING
    elif method == "SIFT":
        detector = cv.SIFT_create(nfeatures=max_features)
        norm = cv.NORM_L2
    else:
        raise ValueError("method must be 'ORB' or 'SIFT'")

    kps1, des1 = detector.detectAndCompute(img1, None)
    kps2, des2 = detector.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(kps1) < 4 or len(kps2) < 4:
        raise RuntimeError("Not enough features detected. Try SIFT or adjust parameters.")

    # KNN + Lowe ratio test (0.75 classic)
    bf = cv.BFMatcher(norm, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Keep only top X% best of the "good" matches (per assignment)
    good = sorted(good, key=lambda x: x.distance)
    keep = max(int(len(good) * good_match_percent), 8)  # need >= 8 for robust H
    good = good[:keep]
    print(f"Good matches after filtering: {len(good)}")

    if len(good) < 8:
        raise RuntimeError(f"Too few good matches ({len(good)}). Try SIFT or different params.")

    # Homography (img1 -> img2)
    pts1 = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, ransacReprojThreshold=4.0)
    if H is None:
        raise RuntimeError("Homography failed. Try SIFT or tweak thresholds.")

    h2, w2 = img2.shape[:2]
    aligned_small = cv.warpPerspective(img1, H, (w2, h2))

    # Draw at most 60 matches so the file stays small
    to_draw = good[:60]
    matches_vis = cv.drawMatches(
        img1, kps1, img2, kps2, to_draw, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return aligned_small, matches_vis


# --------------------------- Main ---------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="IKT213 - Assignment 4")
    p.add_argument("--ref", default="reference_img.png", help="reference image path")
    p.add_argument("--align", default="align_this.jpg", help="image to align path")
    p.add_argument("--method", default="ORB", choices=["ORB", "SIFT"], help="feature method")
    p.add_argument("--max_features", type=int, default=1500, help="ORB:1500, SIFT:10")
    p.add_argument("--good", type=float, default=0.15, help="ORB:0.15, SIFT:0.7")
    args = p.parse_args()

    # Read inputs from same folder as this file
    ref = cv.imread(_here(args.ref), cv.IMREAD_COLOR)
    if ref is None:
        raise FileNotFoundError(f"Missing file: {args.ref} (place next to main.py).")
    to_align = cv.imread(_here(args.align), cv.IMREAD_COLOR)
    if to_align is None:
        raise FileNotFoundError(f"Missing file: {args.align} (place next to main.py).")

    # 1) Harris (page 1)
    harris_img = harris_corners(ref)
    _save_png("01_harris.png", harris_img)

    # 2) Alignment (pages 2 & 3)
    aligned, matches = align_feature_based(
        image_to_align_bgr=to_align,
        reference_bgr=ref,
        method=args.method,
        max_features=args.max_features,
        good_match_percent=args.good,
    )
    _save_png("02_aligned.png", aligned)

    # Save matches as PNG and also JPG (viewer fallback)
    _save_png("03_matches.png", matches)
    jpg_path = _here("03_matches.jpg")
    cv.imwrite(jpg_path, matches, [int(cv.IMWRITE_JPEG_QUALITY), 95])
    print(f"Also wrote 03_matches.jpg -> {jpg_path}")

    print("Done. Combine 01_harris.png, 02_aligned.png, 03_matches.png (or .jpg) into a PDF.")


if __name__ == "__main__":
    main()
