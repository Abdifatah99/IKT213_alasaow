import os
import cv2 as cv
import numpy as np


# -----------------------------
# II. Functions (with comments)
# -----------------------------

def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Detect edges using Sobel after a small Gaussian blur.
    - Blur: ksize=(3,3), sigmaX=0 (assignment requirement)
    - Sobel: dx=1, dy=1, ksize=1 (mixed derivative)
    Returns an 8-bit edge image.
    """
    # Convert to gray for more stable gradients
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)

    # Sobel with 64F to keep negative gradients, then convert to 8-bit
    sob = cv.Sobel(blurred, cv.CV_64F, dx=1, dy=1, ksize=1)
    sob = np.absolute(sob)
    sob = np.uint8(np.clip(sob, 0, 255))
    return sob


def canny_edge_detection(image: np.ndarray, threshold_1: int, threshold_2: int) -> np.ndarray:
    """
    Canny edges after a small Gaussian blur.
    - Blur: ksize=(3,3), sigmaX=0
    - Canny: thresholds provided by caller (50, 50 in the assignment)
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    edges = cv.Canny(blurred, threshold_1, threshold_2)
    return edges


def template_match(image_bgr: np.ndarray, template_bgr: np.ndarray, threshold: float = 0.9) -> tuple[np.ndarray, int]:
    """
    Template matching on GRAYSCALE images, drawing red rectangles on the COLOR copy.
    - Uses TM_CCOEFF_NORMED and marks all locations with score >= threshold.
    Returns (output_bgr, num_matches).
    """
    img_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    tpl_gray = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY)

    res = cv.matchTemplate(img_gray, tpl_gray, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)  # (rows, cols) of top-left corners

    h, w = tpl_gray.shape
    out = image_bgr.copy()
    count = 0
    for pt in zip(*loc[::-1]):  # swap to (x, y)
        cv.rectangle(out, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)  # red rectangle (B,G,R)
        count += 1
    return out, count


def resize(image: np.ndarray, scale_factor: int, up_or_down: str) -> np.ndarray:
    """
    Resize using image pyramids.
    - up_or_down: "up" → pyrUp, "down" → pyrDown
    - scale_factor: how many times to apply the pyramid op (e.g., 2 = twice)
    """
    out = image.copy()
    for _ in range(int(scale_factor)):
        if up_or_down.lower() == "up":
            out = cv.pyrUp(out)
        elif up_or_down.lower() == "down":
            out = cv.pyrDown(out)
        else:
            raise ValueError("up_or_down must be 'up' or 'down'")
    return out


# --------------- Helpers ---------------
def _save(outdir: str, fname: str, img: np.ndarray):
    """Save image and print where it was written + its shape."""
    path = os.path.join(outdir, fname)
    ok = cv.imwrite(path, img)
    if not ok:
        raise IOError(f"Failed to write: {path}")
    print(f"Saved {fname:>25}  shape={img.shape}  ->  {path}")


# ----------------- Main -----------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="IKT213 - Assignment 3")
    parser.add_argument("--lambo", default="lambo.png", help="Path to lambo.png")
    parser.add_argument("--shapes", default="shapes.png", help="Path to shapes.png")
    parser.add_argument("--template", default="shapes_template.jpg", help="Path to shapes_template.jpg")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor for pyramid resize (default=2)")
    parser.add_argument("--outdir", default=".", help="Where to save outputs (default: current folder)")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    outdir = os.path.join(here, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # ---- Read inputs ----
    lambo = cv.imread(os.path.join(here, args.lambo), cv.IMREAD_COLOR)
    if lambo is None:
        raise FileNotFoundError(f"Could not find lambo image at: {os.path.join(here, args.lambo)}")

    shapes = cv.imread(os.path.join(here, args.shapes), cv.IMREAD_COLOR)
    template = cv.imread(os.path.join(here, args.template), cv.IMREAD_COLOR)
    if shapes is None or template is None:
        raise FileNotFoundError(
            "Could not find shapes/template images. Place 'shapes.png' and 'shapes_template.jpg' "
            f"in {here}"
        )

    # ---- Sobel ----
    sob = sobel_edge_detection(lambo)
    _save(outdir, "01_sobel_edges.png", sob)

    # ---- Canny (thresholds 50/50) ----
    can = canny_edge_detection(lambo, 50, 50)
    _save(outdir, "02_canny_edges.png", can)

    # ---- Template Matching (threshold=0.9) ----
    tm_out, n = template_match(shapes, template, threshold=0.9)
    print(f"Template matches found: {n}")
    _save(outdir, "03_template_matches.png", tm_out)

    # ---- Resizing with pyramids ----
    up = resize(lambo, scale_factor=args.scale, up_or_down="up")
    _save(outdir, f"04_resize_up_x{args.scale}.png", up)

    down = resize(lambo, scale_factor=args.scale, up_or_down="down")
    _save(outdir, f"05_resize_down_x{args.scale}.png", down)

    print("All outputs saved to:", outdir)


if __name__ == "__main__":
    main()
