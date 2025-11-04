#!/usr/bin/env python3
"""
multi_folder_open_fronts.py

Detect a single open front curve per image — not a closed loop.
Processes multiple input folders, saving results to a parallel folder tree
under a specified OUTPUT_PARENT_DIR.
"""

import os
import cv2
import numpy as np

# --- CONFIG ---
INPUT_FOLDERS = [
    "./image_processing/fvg/resized/humidity_at_1.4km/",
    "./image_processing/fvg/resized/humidity_at_3km/",
    "./image_processing/fvg/resized/humidity_at_5.5km/",
    "./image_processing/fvg/resized/humidity_at_9km/",
    "./image_processing/fvg/resized/humidity_at_100m/",
    "./image_processing/fvg/resized/humidity_at_750m/"
]

# All results will be saved under this parent folder,
# preserving the same folder names as INPUT_FOLDERS
OUTPUT_PARENT_DIR = "./fronts_curve_open/"

GAUSSIAN_SIGMA_MAIN = 8.0
PERCENTILE = 90
MIN_FRONT_FRACTION = 0.5     # fraction of image diagonal
APPROX_EPSILON = 20.0
LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 3
# ----------------------------


def extract_longest_open_curve(contours):
    """Return a simplified open curve (Nx2) from the longest contour."""
    if not contours:
        return None
    longest = max(contours, key=lambda c: len(c))
    pts = longest[:, 0, :]
    # Find two farthest points to open the shape
    dist = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    if i > j:
        i, j = j, i
    open_curve = pts[i:j + 1]
    if len(open_curve) < 5:
        return None
    simplified = cv2.approxPolyDP(open_curve[:, None, :],
                                  epsilon=APPROX_EPSILON,
                                  closed=False)
    return simplified[:, 0, :]


def detect_single_open_front(image_path, output_dir):
    """Process one image, saving its open front overlay."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"  → {basename}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Cannot open {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    diag = np.hypot(w, h)

    # --- Heavy Gaussian blur before gradient ---
    blurred = cv2.GaussianBlur(gray, (0, 0), GAUSSIAN_SIGMA_MAIN)

    # --- Gradient magnitude ---
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_norm = np.uint8(255 * mag / (np.max(mag) + 1e-8))

    # --- Strong gradient mask ---
    thr_val = np.percentile(mag_norm, PERCENTILE)
    _, edges = cv2.threshold(mag_norm, thr_val, 255, cv2.THRESH_BINARY)

    # --- Post-smoothing ---
    edges = cv2.GaussianBlur(edges, (5, 5), 1.5)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                             iterations=2)

    # --- Contours ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curve = extract_longest_open_curve(contours)
    if curve is None:
        print("     no usable open curve found.")
        return

    length = cv2.arcLength(curve[:, None, :], False)
    if length < MIN_FRONT_FRACTION * diag:
        print(f"     curve too short ({length:.0f}px).")
        return

    # --- Draw open curve ---
    out = cv2.GaussianBlur(img, (0, 0), GAUSSIAN_SIGMA_MAIN / 1.5)
    for i in range(len(curve) - 1):
        cv2.line(out, tuple(curve[i]), tuple(curve[i + 1]),
                 LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{basename}_front.png")
    cv2.imwrite(outpath, out)


def process_folder(input_dir, output_dir):
    """Process all .png images in a single folder."""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
    print(f"\n📁 Processing folder: {input_dir} ({len(files)} images)")
    for f in files:
        try:
            detect_single_open_front(os.path.join(input_dir, f), output_dir)
        except Exception as e:
            print(f"⚠️ Error with {f}: {e}")


def process_multiple_folders(input_folders, output_parent):
    """Mirror folder structure: each input → same-named subfolder under output_parent."""
    os.makedirs(output_parent, exist_ok=True)
    for folder in input_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        output_dir = os.path.join(output_parent, folder_name)
        process_folder(folder, output_dir)


if __name__ == "__main__":
    process_multiple_folders(INPUT_FOLDERS, OUTPUT_PARENT_DIR)
