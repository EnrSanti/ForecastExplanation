import logging
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def batched_kmeans_torch(
    X: np.ndarray,
    num_clusters: int,
    max_iter: int = 200,
    n_init: int = 8,
    tol: float = 1e-4,
    device: str = "cpu",
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized-over-B KMeans for a batch of same-shaped images.
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
        pixel data is identical across restarts, only the centres differ.
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

        # Pre-allocate to avoid repeated allocation in the inner loop
        ones_bp = torch.ones(B, P, device=device)

        for _init in range(n_init):
            idx = torch.randint(0, P, (B, num_clusters), device=device)
            centers = torch.gather(X_t, 1, idx.unsqueeze(-1).expand(-1, -1, C))
            converged = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(max_iter):
                dist = torch.cdist(X_t, centers)  # (B, P, K)
                labels = torch.argmin(dist, dim=-1)  # (B, P)

                new_centers = torch.zeros_like(centers)
                new_centers.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, C), X_t)
                counts = torch.zeros(B, num_clusters, device=device)
                counts.scatter_add_(1, labels, ones_bp)

                empty = (counts == 0).unsqueeze(-1)
                new_centers = new_centers / counts.clamp(min=1).unsqueeze(-1)
                new_centers = torch.where(empty, centers, new_centers)

                # Per-image convergence tracking
                shift = torch.norm(new_centers - centers, dim=(1, 2))  # (B,)
                newly_converged = shift < tol
                # Freeze centers for already-converged images
                centers = torch.where(
                    converged.unsqueeze(-1).unsqueeze(-1), centers, new_centers
                )
                converged = converged | newly_converged
                if converged.all():
                    break

            dist_final = torch.cdist(X_t, centers)
            min_dist, labels = dist_final.min(dim=-1)
            inertia = (min_dist**2).sum(dim=1)  # (B,)

            improved = inertia < best_inertia
            best_inertia = torch.where(improved, inertia, best_inertia)
            best_labels = torch.where(improved.unsqueeze(-1), labels, best_labels)
            best_centers = torch.where(
                improved.unsqueeze(-1).unsqueeze(-1), centers, best_centers
            )

        return best_labels.cpu().numpy(), best_centers.cpu().numpy()


def _order_labels_by_brightness(
    clustering: np.ndarray,
    gray: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
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


_LEGEND_RANGE_RE = re.compile(r"range:\s*([-\d.]+)\s*(\S+)\s*to\s*([-\d.]+)\s*\S+")


def _run_clustering(
    items: List[Tuple[str, np.ndarray]],
    numClusters: int,
    output_dir: str,
    batch_size: int = 8,
    n_init: int = 8,
    max_iter: int = 200,
) -> None:
    """CPU clustering using sklearn KMeans with Elkan's algorithm.

    Elkan's algorithm exploits the triangle inequality to skip most distance
    computations after the first few iterations, and KMeans++ initialization
    converges in far fewer iterations than random init.  Combined with
    sklearn's C/Cython + OpenMP backend this is ~14x faster than the batched
    PyTorch cdist implementation on CPU.
    """
    from sklearn.cluster import KMeans

    for f, img in items:
        if img is None:
            logger.warning(f"Skipping '{f}': image is None.")
            continue

        H, W, C = img.shape
        X = img.reshape(-1, C).astype(np.float32)

        km = KMeans(
            n_clusters=numClusters,
            n_init=n_init,
            max_iter=max_iter,
            algorithm="elkan",
        )
        clustering = km.fit_predict(X).reshape(H, W)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        swapped = _order_labels_by_brightness(clustering, gray, numClusters)

        out_path = os.path.join(output_dir, f)
        cv2.imwrite(out_path, swapped, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


def generate_clustered_images(
    numClusters: int,
    output_dir: str,
    input_dir: Optional[str] = None,
    images_dict: Optional[Dict[str, np.ndarray]] = None,
    legend_dir: Optional[str] = None,
    feature_key: Optional[str] = None,
    batch_size: int = 8,
    n_init: int = 8,
    max_iter: int = 50,
) -> None:
    """
    Generates clustered images either from an input directory or an in-memory dictionary.
    """
    if images_dict is not None:
        items = list(images_dict.items())
    elif input_dir is not None:
        items = []
        for f in os.listdir(input_dir):
            if not f.endswith(".png"):
                continue
            img_path = os.path.join(input_dir, f)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to read image '{img_path}'. Skipping.")
                continue
            items.append((f, img))
    else:
        raise ValueError("Must provide either input_dir or images_dict")

    if torch.cuda.is_available():
        _run_clustering_cuvs(
            items, numClusters, output_dir, n_init=n_init, max_iter=max_iter
        )
    else:
        _run_clustering(items, numClusters, output_dir, batch_size, n_init, max_iter)


def _run_clustering_cuvs(
    items, numClusters, output_dir, n_init=8, max_iter=200, tol=1e-4
):
    """
    One-image-at-a-time cuVS KMeans clustering + brightness-ordered save.
    items : list of (filename, np.ndarray) pairs, e.g. images_dict.items().
    or fallback on cpu
    """
    import cupy as cp
    from cuvs.cluster.kmeans import KMeansParams, fit, predict

    os.makedirs(output_dir, exist_ok=True)
    params = KMeansParams(
        n_clusters=numClusters, max_iter=max_iter, tol=tol, n_init=n_init
    )

    for f, img in items:
        if img is None:
            logger.warning(f"Skipping '{f}': image is None.")
            continue

        H, W, C = img.shape
        X = cp.asarray(img.reshape(-1, C), dtype=cp.float32)  # (P, C) on GPU

        centroids, inertia, n_iter = fit(params, X)
        labels, inertia = predict(params, X, centroids)

        clustering = cp.asnumpy(labels).reshape(H, W).astype(np.int64)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        swapped = _order_labels_by_brightness(clustering, gray, numClusters)

        out_path = os.path.join(output_dir, f)
        cv2.imwrite(out_path, swapped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        del X, centroids, labels
        cp.get_default_memory_pool().free_all_blocks()


def cluster(
    output_dir: str,
    label_dir: str,
    input_dir: Optional[str] = None,
    images_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> None:
    """
    Iterates variable-type subfolders and clusters each with its own K.
    Safe to call from multiple threads on different inputs.
    """
    if images_dict is not None:
        folders = list(images_dict.keys())
    elif input_dir is not None:
        folders = [
            f
            for f in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, f))
        ]
    else:
        raise ValueError("Must provide either input_dir or images_dict")

    for folder in folders:
        output_folder_path = os.path.join(output_dir, folder)
        os.makedirs(output_folder_path, exist_ok=True)

        name = folder.lower()
        if "wind" in name:
            num_clusters = 3
            feature_key = "wind"
        elif "temp" in name:
            num_clusters = 5
            feature_key = "temp"
        elif "cloud" in name:
            num_clusters = 3
            feature_key = "cloud"
        elif "hum" in name:
            num_clusters = 5
            feature_key = "humidity"
        else:
            logger.warning(
                f"Unknown folder type '{folder}'. Skipping clustering for this folder."
            )
            continue

        folder_input_dir = os.path.join(input_dir, folder) if input_dir else None
        folder_images_dict = images_dict[folder] if images_dict else None

        start = time.perf_counter()
        generate_clustered_images(
            numClusters=num_clusters,
            output_dir=output_folder_path,
            input_dir=folder_input_dir,
            images_dict=folder_images_dict,
            legend_dir=label_dir,
            feature_key=feature_key,
        )
        logger.debug(
            f"Finished clustering '{folder}' in {time.perf_counter() - start:.2f} seconds."
        )
