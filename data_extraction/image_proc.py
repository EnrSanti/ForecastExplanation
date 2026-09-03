import logging
from typing import Dict, Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def cluster_xarray(
    data_dict: Dict[str, xr.DataArray],
    num_clusters_map: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """
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
