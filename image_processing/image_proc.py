from PIL import Image, ImageOps
from skimage.measure import label, regionprops
from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
import math, os, time
import cv2
from glob import glob


#----------- CLUSTERING ----------- 
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

def cluster_images(n_im, numClusters, reshaped, image, image_f):
    clustering = [0 for _ in range(n_im)]
    for i in range(n_im):
        kmeans = KMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped[i])
        clustering[i] = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
                                   (image[i].shape[0], image[i].shape[1]))
        print("processing " + image_f[i])

    sortedLabels = [[] for _ in range(n_im)]
    for i in range(n_im):
        # compute mean brightness per cluster
        gray = cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)
        means = []
        for lbl in range(numClusters):
            mask = (clustering[i] == lbl)
            means.append(np.mean(gray[mask]) if np.any(mask) else 0)

        # sort by brightness (dark → bright)
        sortedLabels[i] = sorted(range(numClusters), key=lambda x: means[x])

    kmeansImage = [0 for _ in range(n_im)]
    concatImage = [[] for _ in range(n_im)]
    for j in range(n_im):
        kmeansImage[j] = np.zeros(image[j].shape[:2], dtype=np.uint8)
        for i, label in enumerate(sortedLabels[j]):
            # black = background, gray = border, white = core
            kmeansImage[j][clustering[j] == label] = int((255) / (numClusters - 1)) * i

        concatImage[j] = np.concatenate(
            (image[j],
             193 * np.ones((image[j].shape[0], int(0.0625 * image[j].shape[1]), 3), dtype=np.uint8),
             cv2.cvtColor(kmeansImage[j], cv2.COLOR_GRAY2BGR)),
            axis=1
        )

    return kmeansImage

#----------- IMG RESIZING -----------

def resize_1_4(input_folder, output_folder, scale_factor=0.25):

    # === SETUP ===
    os.makedirs(output_folder, exist_ok=True)

    # Supported image formats
    valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    # === PROCESS IMAGES ===
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        name, ext = os.path.splitext(filename)

        if ext.lower() not in valid_extensions:
            continue  # skip non-image files

        try:
            # open image
            img = Image.open(file_path)

            # compute new size
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))

            # resize with high-quality downsampling
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

            # save to output folder
            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)

            print(f"Resized: {filename} -> {new_size}")

        except Exception as e:
            print(f"Skipping {filename}: {e}")

