from PIL import Image, ImageOps
from skimage.measure import label, regionprops
from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math, os, time
import cv2
from glob import glob

import multiprocessing as mp
import torch

import threading
from concurrent.futures import ThreadPoolExecutor

#----------- CLUSTERING ----------- 


def cluster_images(numClusters, reshaped, image, image_f):
    """
    clustering function of a single image
    
    """
    kmeans = KMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped[i])
    clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
                                   (image[0].shape[0], image[0].shape[1]))
    print("processing " + image_f[i])

    # compute mean brightness per cluster
    gray = cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)
    means = []
    for lbl in range(numClusters):
        mask = (clustering[i] == lbl)
        means.append(np.mean(gray[mask]) if np.any(mask) else 0)

    # sort by brightness (dark → bright)
    sortedLabels = sorted(range(numClusters), key=lambda x: means[x])


    kmeansImage = np.zeros(image[0].shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels[0]):
        # black = background, gray = border, white = core
        kmeansImage[clustering == label] = int((255) / (numClusters - 1)) * i

    concatImage = np.concatenate(
        (image[0],
            193 * np.ones((image[0].shape[0], int(0.0625 * image[0].shape[1]), 3), dtype=np.uint8),
            cv2.cvtColor(kmeansImage, cv2.COLOR_GRAY2BGR)),
        axis=1
    )

    return [kmeansImage]


#----------- IMG RESIZING -----------
def resize_1_4_and_simplify(input_folder, output_folder, scale_factor=0.25):
    """
    function resizing 4 to 1 all the images in the input folder, saves them in the output folder
    
    Parameters
    ----------
        input_folder: the path of the folder where the images to be resized are present
        output_folder: the path of the folder where the images resized are saved
        scale_factor: the scaling factor of the image
    """
    # === SETUP ===
    os.makedirs(output_folder, exist_ok=True)
    #check if output folder contains as much files as input folder
    if len(os.listdir(output_folder)) >= len(os.listdir(input_folder)):
        print(f"Output folder '{output_folder}' already contains images. Skipping resizing.")
        return
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

            #print(f"Resized: {filename} -> {new_size}")

        except Exception as e:
            print(f"Skipping {filename}: {e}")

def kmeans_torch(X, num_clusters=3, max_iter=500,n_init=40, tol=1e-4, device=None):
    """
    GPU-based K-Means using PyTorch.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Input data (pixels/features)
    num_clusters : int
        Number of clusters
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    device : str or torch.device
        "cuda" or "cpu" (auto if None)

    Returns
    -------
    labels : np.ndarray, shape (N,)
        Cluster labels
    centers : np.ndarray, shape (num_clusters, D)
        Cluster centers
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    best_inertia = float("inf")
    best_labels = None
    best_centers = None

    for _ in range(n_init):
        indices = torch.randperm(X_t.shape[0])[:num_clusters]
        centers = X_t[indices]
        for _ in range(max_iter):
            dists = torch.cdist(X_t, centers)
            labels = torch.argmin(dists, dim=1)
            new_centers = torch.stack([
                X_t[labels == k].mean(dim=0) if torch.any(labels == k) else centers[k]
                for k in range(num_clusters)
            ])
            if torch.norm(new_centers - centers) < tol:
                break
            centers = new_centers

        # Compute inertia (sum of squared distances)
        dists_final = torch.cdist(X_t, centers)
        inertia = torch.sum((dists_final[torch.arange(X_t.shape[0]), labels])**2).item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()
            best_centers = centers.clone()

    return best_labels.cpu().numpy(), best_centers.cpu().numpy()

def cluster_images_gpu( numClusters, reshaped, image, image_f):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X = reshaped[0]
    labels, _ = kmeans_torch(X, num_clusters=numClusters, max_iter=200, device=device)
    clustering = labels.reshape(image[0].shape[:2])

    # --- Ensure consistent label mapping ---
    gray = cv2.cvtColor(image[0], cv2.COLOR_BGR2GRAY)
    means = []
    for lbl in range(numClusters):
        mask = (clustering == lbl)
        means.append(np.mean(gray[mask]) if np.any(mask) else 0)
    sortedLabels = sorted(range(numClusters), key=lambda x: means[x])
        
    
    img_mapped = np.zeros_like(clustering, dtype=np.uint8)
    for idx, label in enumerate(sortedLabels):
        img_mapped[clustering == label] = int((255)/(numClusters-1))*idx
    kmeansImage = img_mapped

    return [kmeansImage]

num_workers = 4
_streams = [torch.cuda.Stream(device=0) for _ in range(num_workers)]


def cluster_images_auto(numClusters, reshaped, image, image_f):
    """
    Automatically uses GPU K-Means if CUDA is available.
    Falls back to CPU K-Means otherwise.
    """
    try:
        if torch.cuda.is_available():
            return cluster_images_gpu(numClusters, reshaped, image, image_f)
    except ImportError:
        print("[INFO] PyTorch not installed → using CPU K-Means")

    # fallback to CPU
    return cluster_images(numClusters, reshaped, image, image_f)


def worker_thread(args, results, idx):
    """
    Each thread uses a persistent CUDA stream from _streams
    """
    numClusters, reshaped, image, file, stream_id = args
    stream = _streams[stream_id]

    print(f"[INFO] Thread {threading.get_ident()} using stream {stream}")

    with torch.cuda.stream(stream):
        result = cluster_images_gpu(
            numClusters,
            [reshaped],
            [image],
            [file]
        )[0]

    results[idx] = result


def cluster_images_auto_4threads(numClusters, reshaped_list, image_list, file_list):
    """
    
    Run clustering in batches of 4 threads at a time.
    Each thread uses a distinct CUDA stream.
    """
    total = len(reshaped_list)
    results = [None] * total

    for batch_start in range(0, total, num_workers):
        batch_end = min(batch_start + num_workers, total)

        threads = []

        # Launch up to 4 threads
        for i in range(batch_start, batch_end):
            stream_id = i % num_workers
            t = threading.Thread(
                target=worker_thread,
                args=((numClusters,
                       reshaped_list[i],
                       image_list[i],
                       file_list[i],
                       stream_id), results, i)
            )
            t.start()
            threads.append(t)

        # Wait for this batch to finish before launching the next one
        for t in threads:
            t.join()

    return results



def generate_clustered_images(numClusters, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))
    if len(os.listdir(output_dir)) >= len(files):
        print(f"Output folder '{output_dir}' already contains images. Skipping clustering.")
        return

    batch_files = files[:len(files)]  # e.g., 2 images
    reshaped_batch = []
    images_batch = []

    for f in batch_files:
        img_path = os.path.join(input_dir, f)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Skipping {f}, not a valid image.")
            continue

        H, W, C = img.shape
        reshaped_batch.append(img.reshape(-1, C))
        images_batch.append(img)

    # --- Cluster images in parallel threads using different streams ---
    clustered_results = cluster_images_auto_4threads(
        numClusters,
        reshaped_batch,
        images_batch,
        batch_files
    )

    # --- Save results ---
    for f, clustered_img in zip(batch_files, clustered_results):
        if clustered_img is None:
            continue

        if clustered_img.ndim == 3:
            clustered_gray = cv2.cvtColor(clustered_img, cv2.COLOR_BGR2GRAY)
        else:
            clustered_gray = clustered_img

        unique_vals = np.unique(clustered_gray)
        out_path = os.path.join(output_dir, f)

        if len(unique_vals) == 3:
            unique_vals = np.sort(unique_vals)
            black, mid, white = unique_vals

            swapped = np.zeros_like(clustered_gray)
            swapped[clustered_gray == black] = 0
            swapped[clustered_gray == mid] = 128
            swapped[clustered_gray == white] = 255
        else:
            swapped = clustered_gray

        cv2.imwrite(out_path, swapped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])