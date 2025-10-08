from .imageUtils import recolor, cluster_images
from PIL import Image, ImageOps
from skimage.measure import label, regionprops
from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
import math, os, time
import cv2
from glob import glob


def annotate_clouds_features(input_folder="./clustered", output_folder="./ellipses",
                             min_area=50, iou_thresh=0.7):
    import os, cv2, numpy as np
    from glob import glob

    os.makedirs(output_folder, exist_ok=True)
    img_paths = glob(os.path.join(input_folder, "*.*"))

    def ellipse_overlap(e1, e2):
        x1, y1 = int(e1[0][0] - e1[1][0]/2), int(e1[0][1] - e1[1][1]/2)
        w1, h1 = int(e1[1][0]), int(e1[1][1])
        x2, y2 = int(e2[0][0] - e2[1][0]/2), int(e2[0][1] - e2[1][1]/2)
        w2, h2 = int(e2[1][0]), int(e2[1][1])
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        union_area = w1*h1 + w2*h2 - inter_area
        return inter_area / union_area

    def is_inside(e_small, e_big):
        (cx, cy), (w, h), _ = e_small
        (bx, by), (bw, bh), _ = e_big
        dx, dy = cx - bx, cy - by
        return (dx/(bw/2))**2 + (dy/(bh/2))**2 <= 1

    for path in img_paths:
        img_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_color is None:
            print(f"[WARN] Skipping {path}, cannot read image.")
            continue

        # 1. Create mask for full cloud cores (255)
        if img_color.ndim == 3:
            gray_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img_color
        cloud_mask = (gray_img == 255).astype(np.uint8) * 255

        # 2. Find contours & filter small areas
        contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        # 3. Fit ellipses
        ellipses = [cv2.fitEllipse(c) for c in contours if len(c) >= 5]
        ellipses = sorted(ellipses, key=lambda e: e[1][0]*e[1][1], reverse=True)

        # 4. Filter overlapping ellipses
        filtered_ellipses = []
        for e in ellipses:
            if all(ellipse_overlap(e, f) <= iou_thresh for f in filtered_ellipses) and \
               not any(is_inside(e, f) for f in filtered_ellipses):
                filtered_ellipses.append(e)

        # 5. Draw ellipses on original image
        annotated = img_color.copy()
        for e in filtered_ellipses:
            cv2.ellipse(annotated, e, (0, 0, 255), 2)  # red ellipses

        # 6. Save annotated image
        filename = os.path.basename(path)
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, annotated)
        print(f"[INFO] Processed {filename}, found {len(filtered_ellipses)} cloud cores")

# https://github.com/AbhinavUtkarsh/Image-Segmentation
def generate_clustered_images(numClusters, heatMapDir, clusteredDir):
    import os, cv2
    import numpy as np

    os.makedirs(clusteredDir, exist_ok=True)
    files = os.listdir(heatMapDir)

    for f in files:
        img_path = os.path.join(heatMapDir, f)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Skipping {f}, not a valid image.")
            continue

        H, W, C = img.shape
        reshaped = img.reshape(-1, C)

        # Cluster this single image
        clustered_img = cluster_images(1, numClusters, [reshaped], [img], [f])[0]

        # Convert to grayscale if needed
        if clustered_img.ndim == 3:
            clustered_gray = cv2.cvtColor(clustered_img, cv2.COLOR_BGR2GRAY)
        else:
            clustered_gray = clustered_img

        # Identify unique cluster values
        unique_vals = np.unique(clustered_gray)

        if len(unique_vals) != 3:
            print(f"[WARN] {f}: found {len(unique_vals)} unique clusters, skipping discrete remap.")
            swapped_img = clustered_gray
        else:
            # Sort to ensure consistent order: low → high intensity
            unique_vals = np.sort(unique_vals)
            black_val, mid_val, white_val = unique_vals

            # Map to discrete 0, 128, 255
            swapped_img = np.zeros_like(clustered_gray, dtype=np.uint8)
            swapped_img[clustered_gray == black_val] = 0       # no cloud
            swapped_img[clustered_gray == mid_val] = 128       # thin cloud
            swapped_img[clustered_gray == white_val] = 255     # full cloud

        # Save as high-quality JPEG
        out_path = os.path.join(clusteredDir, f)
        cv2.imwrite(out_path, swapped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        print(f"[INFO] Saved clustered image: {out_path}")



        
def generate_heatMap(imageDir, heatMapDir):
    for f in os.listdir(imageDir):
        print("Processing " + f)
        recolor(imageDir + "/" + f, heatMapDir)