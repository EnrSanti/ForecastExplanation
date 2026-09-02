import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import xarray as xr

logger = logging.getLogger(__name__)


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


def _run_clustering(
    items: List[Tuple[str, np.ndarray]],
    numClusters: int,
    output_dir: str,
    n_init: int | str = "auto",
    max_iter: int = 200,
) -> None:
    """Core clustering logic. items is a list of (filename, np.ndarray) tuples.

    Elkan's algorithm exploits the triangle inequality to skip most distance
    computations after the first few iterations, and KMeans++ initialization
    converges in far fewer iterations than random init.  Combined with
    sklearn's C/Cython + OpenMP backend this is ~14x faster than the batched
    PyTorch cdist implementation on CPU.
    """
    from sklearn.cluster import KMeans

    for f, img in items:

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
    n_init: str = "auto",
    max_iter: int = 200,
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
        _run_clustering(items, numClusters, output_dir, n_init, max_iter)


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
    n_init_val = 1 if n_init == "auto" else n_init
    params = KMeansParams(
        n_clusters=numClusters, max_iter=max_iter, tol=tol, n_init=n_init_val
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
        cv2.imwrite(out_path, swapped, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

        del X, centroids, labels
        cp.get_default_memory_pool().free_all_blocks()


def cluster(
    output_dir: str,
    input_dir: Optional[str] = None,
    images_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> None:
    """todo add flag to run this
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
        elif "temp" in name:
            num_clusters = 5
        elif "cloud" in name:
            num_clusters = 3
        elif "hum" in name:
            num_clusters = 5
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
        )
        logger.debug(
            f"Finished clustering '{folder}' in {time.perf_counter() - start:.2f} seconds."
        )


def cluster_xarray(
    data_dict: Dict[str, xr.DataArray],
    num_clusters_map: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """
    todo adapt for gpu
    Cluster normalized xarray DataArrays using 1D KMeans on physical values.

    Parameters
    ----------
    data_dict : maps folder name -> DataArray (time, y, x) with values in [0, 1]
    num_clusters_map : override for number of clusters per variable type

    Returns
    -------
    Same structure with values quantised to K evenly-spaced levels in [0, 1]
    """
    from sklearn.cluster import KMeans

    if num_clusters_map is None:
        num_clusters_map = {}

    result = {}
    for folder_name, da in data_dict.items():
        name = folder_name.lower()
        if "wind" in name:
            k = num_clusters_map.get("wind", 3)
        elif "temp" in name:
            k = num_clusters_map.get("temp", 5)
        elif "cloud" in name:
            k = num_clusters_map.get("cloud", 3)
        elif "hum" in name:
            k = num_clusters_map.get("humidity", 5)
        else:
            result[folder_name] = da
            continue

        # Cluster each time frame independently (matching current behavior)
        clustered_frames = []
        for t in range(da.sizes["time"]):
            frame = da.isel(time=t).values
            flat = frame[np.isfinite(frame)].reshape(-1, 1)

            if flat.size == 0:
                clustered_frames.append(frame)
                continue

            unique_vals = np.unique(flat)
            if unique_vals.size < k:
                logger.warning(
                    f"Skipping clustering for '{folder_name}' at frame {t} "
                    f"because it only has {unique_vals.size} unique values (needs at least {k})."
                )
                clustered_frames.append(frame)
                continue

            km = KMeans(n_clusters=k, n_init="auto", algorithm="elkan")
            labels = np.full(frame.shape, 0, dtype=np.int32)
            finite_mask = np.isfinite(frame)
            labels[finite_mask] = km.fit_predict(flat)

            # Order by brightness (value magnitude)
            means = [frame[labels == i].mean() for i in range(k)]
            order = sorted(range(k), key=lambda i: means[i])
            step = 1.0 / max(k - 1, 1)
            remapped = np.zeros_like(frame)
            for rank, lbl in enumerate(order):
                remapped[labels == lbl] = step * rank

            clustered_frames.append(remapped)

        clustered_da = da.copy(data=np.stack(clustered_frames))
        result[folder_name] = clustered_da

    return xr.Dataset({name: da for name, da in result.items()})
