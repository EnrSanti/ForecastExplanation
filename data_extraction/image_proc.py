import logging
import os
import threading
from collections import defaultdict
import time

import cv2
import numpy as np
import torch
import re
from contextlib import nullcontext
logger = logging.getLogger(__name__)

DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"





def batched_kmeans_torch(X, num_clusters, max_iter=200, n_init=8, tol=1e-4,
                          device=DEVICE, stream=None):
    """
    Vectorized-over-B KMeans for a batch of same-shaped images, run on the GPU.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor, shape (B, P, C)
        B images, each with P pixels and C channels (already flattened per-image).
    num_clusters : int
    n_init : int
        Random restarts per image. Looped (not vectorized as extra batch
        elements) deliberately: vectorizing restarts requires duplicating the
        full pixel data n_init times (via expand().reshape(), which forces a
        real copy since the expanded tensor is non-contiguous), which blows
        up VRAM by a factor of n_init for little real speed benefit — the
        pixel data is identical across restarts, only the centers differ.
        Looping keeps peak memory proportional to B, not B*n_init.
    stream : torch.cuda.Stream or None
        If given, all GPU work runs on this stream (caller's responsibility to
        create one stream per thread — do not share a stream across threads).

    Returns
    -------
    labels : np.ndarray, shape (B, P)
    centers : np.ndarray, shape (B, num_clusters, C)
    """
    ctx = torch.cuda.stream(stream)
    with ctx:
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        B, P, C = X_t.shape

        best_inertia = torch.full((B,), float("inf"), device=device)
        best_labels = torch.zeros(B, P, dtype=torch.long, device=device)
        best_centers = torch.zeros(B, num_clusters, C, device=device)

        for _init in range(n_init):
            idx = torch.randint(0, P, (B, num_clusters), device=device)
            centers = torch.gather(X_t, 1, idx.unsqueeze(-1).expand(-1, -1, C))

            for _ in range(max_iter):
                dist = torch.cdist(X_t, centers)             # (B, P, K)
                labels = torch.argmin(dist, dim=-1)           # (B, P)

                new_centers = torch.zeros_like(centers)
                new_centers.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, C), X_t)
                counts = torch.zeros(B, num_clusters, device=device)
                counts.scatter_add_(1, labels, torch.ones(B, P, device=device))

                empty = (counts == 0).unsqueeze(-1)
                new_centers = new_centers / counts.clamp(min=1).unsqueeze(-1)
                new_centers = torch.where(empty, centers, new_centers)

                shift = torch.norm(new_centers - centers, dim=(1, 2))
                centers = new_centers
                if torch.max(shift) < tol:
                    break

            dist_final = torch.cdist(X_t, centers)
            min_dist, labels = dist_final.min(dim=-1)
            inertia = (min_dist ** 2).sum(dim=1)               # (B,)

            improved = inertia < best_inertia
            best_inertia = torch.where(improved, inertia, best_inertia)
            best_labels = torch.where(improved.unsqueeze(-1), labels, best_labels)
            best_centers = torch.where(improved.unsqueeze(-1).unsqueeze(-1), centers, best_centers)

        return best_labels.cpu().numpy(), best_centers.cpu().numpy()

def _order_labels_by_brightness(clustering, gray, num_clusters):
    """Remap arbitrary cluster ids to brightness rank (0=darkest .. K-1=brightest)."""
    means = []
    for lbl in range(num_clusters):
        mask = clustering == lbl
        means.append(np.mean(gray[mask]) if np.any(mask) else 0)
    order = sorted(range(num_clusters), key=lambda k: means[k])

    remapped = np.zeros_like(clustering, dtype=np.uint8)
    step = 255 // max(num_clusters - 1, 1)
    for rank, lbl in enumerate(order):
        remapped[clustering == lbl] = step * rank
    return remapped



_LEGEND_RANGE_RE = re.compile(r'range:\s*([-\d.]+)\s*(\S+)\s*to\s*([-\d.]+)\s*\S+')

def _parse_legend_range(legend_dir, feature_key):
    """
    Read legend_{feature_key}.txt (written by create_legends) and return
    (vmin, vmax, unit). Returns (None, None, None) if the file is missing or
    doesn't match the expected 'X range: <vmin> <unit> to <vmax> <unit>' format,
    so callers can fall back to plain gray-level labeling instead of crashing.
    """
    if not legend_dir or not feature_key:
        return None, None, None
 
    txt_path = os.path.join(legend_dir, f"legend_{feature_key}.txt")
    if not os.path.exists(txt_path):
        logger.warning(f"Legend file not found: '{txt_path}'.")
        return None, None, None
 
    with open(txt_path) as f:
        first_line = f.readline()
 
    m = _LEGEND_RANGE_RE.search(first_line)
    if not m:
        logger.warning(f"Could not parse range from '{txt_path}': {first_line!r}")
        return None, None, None
 
    vmin, unit, vmax = m.group(1), m.group(2), m.group(3)
    return float(vmin), float(vmax), unit



def generate_clustered_images(numClusters, input_dir, output_dir, 
                               legend_dir, feature_key,
                               batch_size=8, n_init=8, max_iter=200):

    
    vmin, vmax, unit = _parse_legend_range(legend_dir, feature_key)
    if vmin is None or vmax is None or unit is None:
        logger.error(f"Legend range not found for feature '{feature_key}'.")
        return
 
    files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
 
    # Group by shape so each GPU batch has a uniform pixel count.
    groups = defaultdict(list)  # (H, W, C) -> [(filename, img)]
    for f in files:
        img_path = os.path.join(input_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to read image '{img_path}'. Skipping.")
            continue
        groups[img.shape].append((f, img))


    device = "cuda" if torch.cuda.is_available() else "cpu"
    stream = torch.cuda.Stream(device=0) if device == "cuda" else None

    for shape, items in groups.items():
        H, W, C = shape
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            batch_files = [f for f, _ in chunk]
            batch_imgs = [img for _, img in chunk]

            logger.debug(time.time(), f"Clustering {len(batch_imgs)} images of shape {shape} from '{input_dir}' -> '{output_dir}'")

            X = np.stack([img.reshape(-1, C) for img in batch_imgs], axis=0)  # (B, P, C)
            labels, _ = batched_kmeans_torch(
                X, numClusters, max_iter=max_iter, n_init=n_init,
                device=DEVICE, stream=stream,
            )

            logger.debug(time.time(), f"Finished clustering {len(batch_imgs)} images of shape {shape} from '{input_dir}' -> '{output_dir}'")

            for f, img, lbl in zip(batch_files, batch_imgs, labels):
                clustering = lbl.reshape(H, W)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                swapped = _order_labels_by_brightness(clustering, gray, numClusters)

                out_path = os.path.join(output_dir, f)
                cv2.imwrite(out_path, swapped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            logger.debug(time.time(), f"Saved clustered images to '{output_dir}'")

            del X, labels, batch_imgs, chunk
            if device == "cuda":
                torch.cuda.synchronize(stream)
                torch.cuda.empty_cache()    


def cluster(input_dir, output_dir, label_dir):
    """
    Iterates variable-type subfolders and clusters each with its own K.
    Safe to call from multiple threads on different (input_dir, output_dir,
    label_dir) triples — each call only touches its own folders, and the
    output-folder "already done" check is lock-protected per output_dir.
    """
    


    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        output_folder_path = os.path.join(output_dir, folder)
        os.makedirs(output_folder_path, exist_ok=True)

        name = folder.lower()
        if "wind" in name and False:
            generate_clustered_images(3, folder_path, output_folder_path, label_dir, feature_key="wind")
        elif "temp" in name:
            generate_clustered_images(5, folder_path, output_folder_path, label_dir, feature_key="temp")
        elif "cloud" in name:
            generate_clustered_images(3, folder_path, output_folder_path, label_dir, feature_key="cloud")
        elif "hum" in name:
            generate_clustered_images(5, folder_path, output_folder_path, label_dir, feature_key="humidity")
        else:
            logger.warning(f"Unknown folder type '{folder}'. Skipping clustering for this folder.")