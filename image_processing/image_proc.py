from PIL import Image, ImageOps
from skimage.measure import label, regionprops
from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math, os, time
import cv2
from glob import glob
from PIL import Image
from skimage import morphology, measure
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

#----------- CLUSTERING ----------- 
# https://github.com/AbhinavUtkarsh/Image-Segmentation
def generate_clustered_images(numClusters, input_dir, output_dir):
    import os, cv2
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)
    if len(os.listdir(output_dir)) >= len(os.listdir(input_dir)):
        print(f"Output folder '{output_dir}' already contains images. Skipping clustering, assuming to be correct.")
        return
    for f in files:
        img_path = os.path.join(input_dir, f)
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
        out_path = os.path.join(output_dir, f)
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


def resize_1_4_and_simplify(input_folder, output_folder, scale_factor=0.25,
                               blur_sigma=1.5, morph_radius=2, simplify_tolerance=3.0):
    """
    Resize cloud images and simplify their shapes by merging small blobs and smoothing edges.

    Parameters
    ----------
    input_folder : str
        Path with original cloud images.
    output_folder : str
        Path to save resized & simplified outputs.
    scale_factor : float, optional
        Downscale factor (default 0.25).
    blur_sigma : float, optional
        Gaussian blur sigma for smoothing clouds before thresholding.
    morph_radius : int, optional
        Morphological radius to merge close blobs and fill small holes.
    simplify_tolerance : float, optional
        Polygon simplification tolerance (higher = smoother clouds).
    """

    os.makedirs(output_folder, exist_ok=True)
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    for fname in os.listdir(input_folder):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in valid_exts:
            continue

        input_path = os.path.join(input_folder, fname)
        output_path = os.path.join(output_folder, fname)

        try:
            # === Step 1: load and resize ===
            img = Image.open(input_path).convert("L")
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            img_arr = np.array(img_resized, dtype=np.float32)

            # === Step 2: Gaussian smoothing ===
            blurred = cv2.GaussianBlur(img_arr, (0, 0), blur_sigma)

            # === Step 3: Normalize + adaptive threshold ===
            norm = cv2.normalize(blurred, None, 0, 1, cv2.NORM_MINMAX)
            thresh_val = np.percentile(norm, 70)  # keep top 30% of bright (cloudy) pixels
            binary = norm > thresh_val

            # === Step 4: Morphological smoothing ===
            selem = morphology.disk(morph_radius)
            binary = morphology.binary_closing(binary, selem)
            binary = morphology.binary_opening(binary, selem)
            binary = morphology.remove_small_objects(binary, 10)

            # === Step 5: Optional shape simplification ===
            contours = measure.find_contours(binary, 0.5)
            polys = []
            for c in contours:
                if len(c) < 4:
                    continue
                p = Polygon(c)
                if p.is_valid and p.area > 4:
                    polys.append(p.simplify(simplify_tolerance))

            if polys:
                merged = unary_union(polys)
                mask = np.zeros_like(binary, dtype=bool)

                if isinstance(merged, Polygon):
                    merged = [merged]
                elif isinstance(merged, MultiPolygon):
                    merged = list(merged.geoms)

                for p in merged:
                    if not p.is_valid or p.area < 1:
                        continue
                    rr, cc = np.round(np.array(p.exterior.coords.xy)).astype(int)
                    rr = np.clip(rr, 0, mask.shape[1] - 1)
                    cc = np.clip(cc, 0, mask.shape[0] - 1)
                    mask[cc, rr] = True
            else:
                mask = binary

            # === Step 6: Save ===
            out_img = Image.fromarray((mask * 255).astype(np.uint8))
            out_img.save(output_path)
            print(f"Processed: {fname} -> {new_size}")

        except Exception as e:
            print(f"Skipping {fname}: {e}")