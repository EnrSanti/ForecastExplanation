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
                         tol=0, min_area=50, iou_thresh=0.7):
    os.makedirs(output_folder, exist_ok=True)
    # Find all images in folder (jpg, png)
    img_paths = glob(os.path.join(input_folder, "*.*"))
    
    def ellipse_overlap(e1, e2):
        # Get bounding rectangles
        x1, y1 = int(e1[0][0] - e1[1][0]/2), int(e1[0][1] - e1[1][1]/2)
        w1, h1 = int(e1[1][0]), int(e1[1][1])
        
        x2, y2 = int(e2[0][0] - e2[1][0]/2), int(e2[0][1] - e2[1][1]/2)
        w2, h2 = int(e2[1][0]), int(e2[1][1])
        
        # Compute intersection rectangle
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0  # no overlap
        
        # Compute areas
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        union_area = w1*h1 + w2*h2 - inter_area
        return inter_area / union_area
    
    def is_inside(e_small, e_big):
        """Check if the center of e_small is inside e_big"""
        (cx, cy), (w, h), angle = e_small
        (bx, by), (bw, bh), bangle = e_big
        
        # Normalize coordinates relative to big ellipse
        dx = cx - bx
        dy = cy - by
        # Approximate ellipse equation: (dx/w/2)^2 + (dy/h/2)^2 <= 1
        if (dx/(bw/2))**2 + (dy/(bh/2))**2 <= 1:
            return True
        return False

    for path in img_paths:
        # 1. Read original image
        img_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_color is None:
            print(f"Skipping {path}, cannot read image.")
            continue
        
        img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

        # 2. Mask nearly-white clouds (254,254,254)
        cloud_mask = np.all(np.abs(img_rgb - 254) <= tol, axis=2).astype(np.uint8) * 255

        # 3. Find contours & filter tiny noise
        contours_high, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_high = [c for c in contours_high if cv2.contourArea(c) >= min_area]

        # 4. Fit ellipses
        ellipses = [cv2.fitEllipse(c) for c in contours_high if len(c) >= 5]
        ellipses = sorted(ellipses, key=lambda e: e[1][0]*e[1][1], reverse=True)

        # 5. Filter overlapping ellipses
        filtered_ellipses = []
        for e in ellipses:
            if all((ellipse_overlap(e, f) <= iou_thresh) for f in filtered_ellipses) and not any(is_inside(e, f) for f in filtered_ellipses):
                filtered_ellipses.append(e)

        # 6. Draw ellipses on original image
        annotated = img_color.copy()
        for e in filtered_ellipses:
            cv2.ellipse(annotated, e, (0, 0, 255), 2)  # red ellipses

        # 7. Save annotated image
        filename = os.path.basename(path)
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, annotated)
        print(f"Processed {filename}, found {len(filtered_ellipses)} clouds/relevant features")

# https://github.com/AbhinavUtkarsh/Image-Segmentation
def generate_clustered_images(numClusters, heatMapDir, clusteredDir):
    import os, time, cv2

    files = os.listdir(heatMapDir)

    for f in files:
        img = cv2.imread(os.path.join(heatMapDir, f))
        H, W, C = img.shape
        reshaped = img.reshape(H*W, C)

        # Cluster this single image
        clustered_img = cluster_images(1, numClusters, [reshaped], [img], [f])[0]

        # Write output
        filename = os.path.join(clusteredDir, f)
        cv2.imwrite(filename, clustered_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
def generate_heatMap(imageDir, heatMapDir):
    for f in os.listdir(imageDir):
        print("Processing " + f)
        recolor(imageDir + "/" + f, heatMapDir)