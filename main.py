"""
main.py
=======
Entry point for the drone terrain mapping pipeline.

Runs the full SIFT-based image stitching pipeline on two drone footage
frames and produces a cropped panoramic terrain map.

Usage:
    python main.py

Update IMAGE_LEFT and IMAGE_RIGHT below to point to your input frames.
Place input images in the same directory or provide full paths.
"""

import cv2
from stitcher import detect_keypoints, match_features, compute_homography, warp_and_stitch, draw_overlap_region
from utils import load_image, show, save, trim, draw_keypoints, draw_matches


# ---------------------------------------------------------------------------
# Configuration — update these paths to your drone footage frames
# ---------------------------------------------------------------------------
IMAGE_LEFT   = 'LEFTLEFT.jpg'
IMAGE_RIGHT  = 'RIGHTRIGHT.jpg'
OUTPUT_PATH  = 'results/terrain_panorama.jpg'


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    print("\n=== Drone Terrain Mapping — Image Stitching Pipeline ===\n")

    # Step 1 — Load images
    print("[STEP 1] Loading images...")
    img_left,  gray_left  = load_image(IMAGE_LEFT)
    img_right, gray_right = load_image(IMAGE_RIGHT)

    # Step 2 — Detect SIFT keypoints and descriptors
    print("\n[STEP 2] Detecting keypoints with SIFT...")
    kp1, des1 = detect_keypoints(gray_left)
    kp2, des2 = detect_keypoints(gray_right)

    show('Left Image — Keypoints', draw_keypoints(img_left, kp1))

    # Step 3 — Match features using FLANN
    print("\n[STEP 3] Matching features with FLANN...")
    good = match_features(des1, des2)

    show('Feature Matches', draw_matches(img_left, kp1, img_right, kp2, good))

    # Step 4 — Compute homography
    print("\n[STEP 4] Computing homography with RANSAC...")
    M, mask = compute_homography(kp1, kp2, good)

    if M is None:
        print("[ERROR] Homography failed. Exiting.")
        return

    # Show overlapping region
    overlap = draw_overlap_region(gray_right, kp1, M, gray_left.shape)
    show('Overlapping Region', overlap)

    # Step 5 — Warp and stitch
    print("\n[STEP 5] Warping and stitching frames...")
    stitched = warp_and_stitch(img_left, img_right, M)
    show('Stitched Panorama (uncropped)', stitched)

    # Step 6 — Trim black borders
    print("\n[STEP 6] Trimming black borders...")
    panorama = trim(stitched)
    show('Terrain Panorama (final)', panorama)

    # Save output
    save(OUTPUT_PATH, panorama)
    print(f"\n=== Done. Panorama saved to {OUTPUT_PATH} ===\n")


if __name__ == '__main__':
    main()
