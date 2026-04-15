"""
utils.py
========
Utility functions for image loading, display, and post-processing.
"""

import cv2
import numpy as np


def load_image(path: str):
    """
    Load an image from disk and return both BGR and grayscale versions.

    Args:
        path : File path to the image.

    Returns:
        img_bgr  : Original BGR image as NumPy array.
        img_gray : Grayscale version for feature detection.
    """
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"[ERROR] Could not load image: {path}")
    img_bgr  = cv2.resize(img_bgr, (0, 0), fx=1, fy=1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(f"[INFO] Loaded: {path} — {img_bgr.shape[1]}×{img_bgr.shape[0]} px")
    return img_bgr, img_gray


def show(title: str, image: np.ndarray):
    """
    Display an image in a named window and wait for a key press.

    Args:
        title : Window title string.
        image : Image to display as NumPy array.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save(path: str, image: np.ndarray):
    """
    Save an image to disk.

    Args:
        path  : Output file path (e.g. 'results/panorama.jpg').
        image : Image to save as NumPy array.
    """
    cv2.imwrite(path, image)
    print(f"[INFO] Saved: {path}")


def trim(frame: np.ndarray) -> np.ndarray:
    """
    Remove black border rows and columns from a stitched panoramic image.

    After perspective warping, the canvas contains black (zero-value) regions
    outside the stitched area. This function recursively crops all four edges
    until no all-zero rows or columns remain.

    Args:
        frame : Input stitched image as NumPy array.

    Returns:
        Cropped image with black borders removed.
    """
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def draw_keypoints(img_bgr: np.ndarray, keypoints) -> np.ndarray:
    """
    Draw detected SIFT keypoints on an image.

    Args:
        img_bgr    : Original BGR image.
        keypoints  : List of cv2.KeyPoint objects.

    Returns:
        Image with keypoints drawn.
    """
    return cv2.drawKeypoints(img_bgr, keypoints, None)


def draw_matches(img1, kp1, img2, kp2, good: list) -> np.ndarray:
    """
    Draw matched keypoints between two images with green connecting lines.

    Args:
        img1, img2 : BGR images.
        kp1, kp2   : Keypoints for each image.
        good       : List of valid cv2.DMatch objects.

    Returns:
        Side-by-side image with matches drawn in green.
    """
    draw_params = dict(
        matchColor       = (0, 255, 0),
        singlePointColor = None,
        flags            = 2
    )
    return cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
