"""
stitcher.py
===========
Core image stitching logic — SIFT keypoint detection, FLANN feature
matching, homography estimation, and perspective warping.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MIN_MATCH_COUNT = 10      # Minimum valid matches required to attempt stitching
LOWE_RATIO      = 0.03    # Lowe's ratio test threshold
FLANN_TREES     = 5       # Number of randomised KD-Trees
FLANN_CHECKS    = 50      # Number of recursive traversals per search


def detect_keypoints(image: np.ndarray):
    """
    Detect keypoints and compute descriptors using SIFT.

    SIFT builds a Difference of Gaussians (DoG) scale-space pyramid,
    localises keypoints at extrema, and generates a 128-dimensional
    gradient descriptor per keypoint — invariant to scale and rotation.

    Args:
        image : Grayscale input image as NumPy array.

    Returns:
        keypoints   : List of cv2.KeyPoint objects.
        descriptors : NumPy array of shape (N, 128).
    """
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    print(f"[INFO] Keypoints detected: {len(keypoints)}")
    return keypoints, descriptors


def match_features(des1: np.ndarray, des2: np.ndarray):
    """
    Match descriptors between two images using FLANN with Lowe's ratio test.

    FLANN uses randomised KD-Tree indexing for fast approximate nearest
    neighbour search. Lowe's ratio test retains only unambiguous matches
    where the best match is significantly closer than the second best.

    Args:
        des1 : Descriptors from image 1 — shape (N, 128).
        des2 : Descriptors from image 2 — shape (M, 128).

    Returns:
        good : List of reliable cv2.DMatch objects.
    """
    index_params  = dict(algorithm=0, trees=FLANN_TREES)
    search_params = dict(checks=FLANN_CHECKS)

    flann   = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
    print(f"[INFO] Valid matches after ratio test: {len(good)}")
    return good


def compute_homography(kp1, kp2, good: list):
    """
    Estimate the homography matrix between two frames using RANSAC.

    Homography maps points from the coordinate space of image 1 into
    image 2, enabling perspective warp for alignment.

    Args:
        kp1  : Keypoints from image 1.
        kp2  : Keypoints from image 2.
        good : List of valid matches.

    Returns:
        M    : 3×3 homography matrix, or None if estimation fails.
        mask : Inlier mask from RANSAC.
    """
    if len(good) < MIN_MATCH_COUNT:
        print(f"[ERROR] Not enough matches: {len(good)} / {MIN_MATCH_COUNT}")
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(f"[INFO] Homography inliers: {int(mask.ravel().sum())} / {len(good)}")
    return M, mask


def warp_and_stitch(img_left: np.ndarray, img_right: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Warp the left image into the right image's perspective and stitch.

    Warps img_left using the homography matrix M onto a wide canvas,
    then composites img_right on top to produce the panorama.

    Args:
        img_left  : Left BGR frame.
        img_right : Right BGR frame.
        M         : 3×3 homography matrix.

    Returns:
        stitched : Uncropped panoramic image as NumPy array.
    """
    panorama_width = img_right.shape[1] + img_left.shape[1]
    stitched = cv2.warpPerspective(img_left, M, (panorama_width, img_right.shape[0]))
    stitched[0:img_right.shape[0], 0:img_right.shape[1]] = img_right
    return stitched


def draw_overlap_region(gray_right: np.ndarray, kp1, M: np.ndarray, shape) -> np.ndarray:
    """
    Draw the overlapping region boundary on the right image for visualisation.

    Args:
        gray_right : Grayscale right image.
        kp1        : Keypoints from left image.
        M          : Homography matrix.
        shape      : (height, width) of left image.

    Returns:
        img_overlap : Right image with overlap boundary drawn.
    """
    h, w = shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img_overlap = cv2.polylines(gray_right.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    return img_overlap
