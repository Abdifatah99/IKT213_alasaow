import os
import cv2 as cv
import numpy as np


# -------------------------------
# II. Functions
# -------------------------------

def padding(image, border_width: int):
    """
    Add a reflective border around the image.

    Parameters
    ----------
    image : np.ndarray (H, W, 3) uint8, BGR order (OpenCV default)
    border_width : int
        Number of pixels to add on each side.

    Returns
    -------
    np.ndarray
        Padded image with shape (H + 2*border_width, W + 2*border_width, 3)
    """
    return cv.copyMakeBorder(
        image,
        border_width, border_width, border_width, border_width,
        borderType=cv.BORDER_REFLECT
    )


def crop(image, x_0: int, x_1: int, y_0: int, y_1: int):
    """
    Crop a rectangular region from the image using array slicing.

    Note: In OpenCV/Numpy, the first index is rows (y), second is columns (x).

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    return image[y_0:y_1, x_0:x_1]


def resize(image, width: int, height: int):
    """
    Resize the image to (width x height) using bilinear interpolation.
    """
    return cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)


def copy(image, emptyPictureArray: np.ndarray):
    """
    Manually copy pixels from `image` into `emptyPictureArray`.

    Constraints: Do NOT use cv2.copy(); show loops/array indexing instead.
    The destination must have the same (H, W, 3) and dtype=uint8.
    """
    h, w, _ = image.shape
    assert emptyPictureArray.shape == image.shape and emptyPictureArray.dtype == np.uint8
    for i in range(h):
        for j in range(w):
            emptyPictureArray[i, j] = image[i, j]
    return emptyPictureArray


def grayscale(image):
    """
    Convert BGR color image to a single-channel grayscale image.
    """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def hsv(image):
    """
    Convert BGR image to HSV color space.

    Warning: Saving an HSV array with cv.imwrite() will produce an image
    whose colors look odd in normal viewers (they expect BGR/RGB). This is
    still valid for the exercise since we are asked to 'convert to HSV' and save.
    """
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


def hue_shifted(image, emptyPictureArray: np.ndarray, hue: int):
    """
    Shift ALL color channel values by `hue` (e.g., +50) with clipping to [0, 255].

    This follows the assignment text: treat the image as a 3D array and change
    the color values directly (not the HSV hue angle). We perform safe clipping
    so values never go below 0 or above 255.

    Parameters
    ----------
    image : np.ndarray uint8
    emptyPictureArray : np.ndarray uint8
        Will receive the shifted result (resized if shape mismatch).
    hue : int
        Additive shift to apply to all three channels.

    Returns
    -------
    np.ndarray
        Shifted BGR image.
    """
    if emptyPictureArray.shape != image.shape or emptyPictureArray.dtype != np.uint8:
        emptyPictureArray = np.zeros_like(image, dtype=np.uint8)

    # Convert to a larger type to avoid overflow, add, then clip back to [0, 255]
    shifted = image.astype(np.int16) + int(hue)
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)

    emptyPictureArray[:] = shifted
    return emptyPictureArray


def smoothing(image):
    """
    Apply Gaussian blur with kernel size 15x15 and default border handling.
    """
    return cv.GaussianBlur(image, (15, 15), 0, borderType=cv.BORDER_DEFAULT)


def rotation(image, rotation_angle: int):
    """
    Rotate the image by 90° clockwise or 180°.

    Parameters
    ----------
    rotation_angle : int
        90  -> rotate 90 degrees clockwise
        180 -> rotate 180 degrees

    Returns
    -------
    np.ndarray
        Rotated image.
    """
    if rotation_angle == 90:
        return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    if rotation_angle == 180:
        return cv.rotate(image, cv.ROTATE_180)
    raise ValueError("rotation_angle must be 90 or 180")


# -------------------------------
# Helper
# -------------------------------
def _save(outdir: str, fname: str, img: np.ndarray):
    """Save an image in `outdir` and print where it went + its shape."""
    path = os.path.join(outdir, fname)
    cv.imwrite(path, img)
    print(f"Saved {fname:>24}  shape={img.shape}  ->  {path}")
    return path


# -------------------------------
# Main runner
# -------------------------------
def main():
    # Work inside the same folder as this script (solutions/)
    here = os.path.dirname(__file__)
    img_path = os.path.join(here, "lena.png")

    # Read input image (BGR, 8-bit)
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(
            f"Could not find image: {img_path}. "
            f"Place 'lena.png' in this folder."
        )

    print("Input image shape (H, W, C):", img.shape)

    # 1) Padding (reflect, border_width = 100)
    pad = padding(img, border_width=100)
    _save(here, "01_padding_reflect.png", pad)

    # 2) Cropping (80 from left/top, 130 from right/bottom)
    h, w = img.shape[:2]
    cr = crop(img, 80, w - 130, 80, h - 130)
    _save(here, "02_crop.png", cr)

    # 3) Resize to 200 x 200
    rz = resize(img, 200, 200)
    _save(here, "03_resize_200x200.png", rz)

    # 4) Manual copy (no cv2.copy)
    empty = np.zeros_like(img, dtype=np.uint8)
    cp = copy(img, empty)
    _save(here, "04_copy_manual.png", cp)

    # 5) Grayscale
    gray = grayscale(img)
    _save(here, "05_grayscale.png", gray)

    # 6) HSV conversion
    hsv_img = hsv(img)
    _save(here, "06_hsv.png", hsv_img)

    # 7) Color shift by +50 on all channels (with clipping)
    empty2 = np.zeros_like(img, dtype=np.uint8)
    shifted = hue_shifted(img, empty2, hue=50)
    _save(here, "07_hue_shift_plus50.png", shifted)

    # 8) Smoothing (Gaussian blur 15x15)
    blur = smoothing(img)
    _save(here, "08_smoothing_gaussian15.png", blur)

    # 9) Rotations
    rot90 = rotation(img, 90)
    _save(here, "09_rotate_90.png", rot90)

    rot180 = rotation(img, 180)
    _save(here, "10_rotate_180.png", rot180)

    print("All outputs written to:", here)


if __name__ == "__main__":
    main()
