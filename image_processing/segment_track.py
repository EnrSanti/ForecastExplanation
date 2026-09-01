import logging
import warnings
from typing import Dict, List, Optional, Set, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
import tobac
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from image_processing.constants import (
    DEFAULT_DT,
    DEFAULT_DXY,
    DEFAULT_GAP_FRAMES,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_SMOOTH,
)

logger = logging.getLogger(__name__)
logging.getLogger("trackpy").setLevel(logging.WARNING)

DEBUG = False  # Set to True to display search radius circles


def detect_features(
    data_norm: xr.DataArray,
    threshold: float,
    target: str,
    smooth: float = DEFAULT_SMOOTH,
    min_blob_size: int = 100,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    dxy: float = DEFAULT_DXY,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs multithreshold feature detection on normalized DataArray.
    Returns (features for segmentation, features_weighted_points for tracking).
    """
    features = tobac.feature_detection_multithreshold(
        data_norm,
        threshold=[threshold],
        dxy=dxy,
        target=target,
        position_threshold="extreme",
        sigma_threshold=smooth,
        n_min_threshold=min_blob_size,
        min_distance=min_distance,
    )
    features_weighted_points = tobac.feature_detection_multithreshold(
        data_norm,
        threshold=[threshold],
        dxy=dxy,
        target=target,
        position_threshold="weighted_abs",
        sigma_threshold=smooth,
        n_min_threshold=min_blob_size,
        min_distance=min_distance,
    )
    return features, features_weighted_points


def track_features(
    features_weighted_points: pd.DataFrame,
    referenced_data: xr.DataArray,
    v_max: float,
    dt: float = DEFAULT_DT,
    dxy: float = DEFAULT_DXY,
    memory: int = DEFAULT_GAP_FRAMES,
    method_linking: str = "predict",
) -> pd.DataFrame:
    """Links detected features into trajectories across time frames."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Could not generate velocity field for prediction: no tracks",
            )
            trajectories = tobac.linking_trackpy(
                features_weighted_points,
                referenced_data,
                dt=dt,
                dxy=dxy,
                v_max=v_max,
                memory=memory,
                method_linking=method_linking,
            )
        return trajectories
    except Exception as e:
        logger.debug(f"No trajectories found: {e}")
        return pd.DataFrame()


def segment_features(
    features: pd.DataFrame,
    data_norm: xr.DataArray,
    threshold: float,
    target: str,
    smooth: float = DEFAULT_SMOOTH,
    dxy: float = DEFAULT_DXY,
) -> Tuple[List[Tuple[int, Optional[xr.DataArray], Optional[xr.DataArray]]]]:
    """Performs 2D segmentation for each frame in data_norm."""
    segments_all = []
    images_no = len(data_norm.time)

    for itime in range(images_no):
        smoothed_frame = cv2.GaussianBlur(
            data_norm.isel(time=itime).values, (0, 0), sigmaX=smooth, sigmaY=smooth
        )
        temp_da = data_norm.isel(time=[itime]).copy()
        temp_da.data = smoothed_frame[np.newaxis, ...]

        f = (
            features[features["frame"] == itime]
            if features is not None
            else pd.DataFrame()
        )
        if f.empty:
            segments_all.append((itime, None, None))
            continue

        segment_labels, segments = tobac.segmentation_2D(
            f,
            temp_da,
            dxy=dxy,
            threshold=threshold,
            target=target,
        )
        segments_all.append((itime, segment_labels, segments))

    return segments_all


def print_clouds_center_line(
    printing_symbol: str,
    color: str,
    f_weighted: pd.DataFrame,
    itime: int,
    track: pd.DataFrame,
    axs: plt.Axes,
    cell_id: int,
    persisted_cells: Set[int],
    all_frames_for_cell: Dict[int, List[int]],
):
    """Plots cloud center markers and fading trajectory path on axes."""
    cell_in_this_frame = not (track[track["frame"] == itime].empty)

    if cell_id in persisted_cells and cell_in_this_frame:
        frames_list = all_frames_for_cell.get(int(cell_id), [])
        if frames_list:
            last_frame = frames_list[-1]
            line = track[(track["frame"] == last_frame) | (track["frame"] == itime)]
            axs.plot(line["x"], line["y"], color="blue", linewidth=1.5, alpha=0.5)

    try:
        frames = all_frames_for_cell.get(int(cell_id), [])
        for t0, t1 in zip(frames[:-1], frames[1:]):
            line = track[(track["frame"] == t0) | (track["frame"] == t1)]
            time_diff = itime - track.iloc[0].frame
            alpha = 0.1 + 0.3 * (t0 - track.iloc[0].frame) / (
                time_diff if time_diff != 0 else 1
            )
            alpha = max(0.05, min(1.0, alpha))
            axs.plot(line["x"], line["y"], color="blue", linewidth=1.5, alpha=alpha)
    except (KeyError, IndexError) as e:
        logger.debug(f"Failed to draw trail for cell {cell_id}: {e}")

    if not f_weighted.empty:
        axs.scatter(
            f_weighted["x"],
            f_weighted["y"],
            s=40,
            color=color,
            marker=printing_symbol,
            zorder=5,
        )


def print_cloud_labels(
    f_weighted: pd.DataFrame,
    cell_id: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    axs: plt.Axes,
):
    """Renders cell ID text annotation on axes."""
    if f_weighted.empty or "x" not in f_weighted or len(f_weighted["x"]) == 0:
        return

    x_pos = f_weighted["x"].values[0]
    y_pos = f_weighted["y"].values[0]

    if x_pos < xlim[0] + 30:
        x_pos = x_pos + 20
    if x_pos > xlim[1] - 30:
        x_pos = x_pos - 20

    if y_pos < ylim[0] + 30:
        y_pos = y_pos + 20
    if y_pos > ylim[1] - 30:
        y_pos = y_pos - 20

    axs.text(
        x_pos - 3,
        y_pos - 3,
        f"{int(cell_id)}",
        color="white",
        fontsize=8,
        weight="bold",
        bbox=dict(facecolor="black", alpha=0.3, edgecolor="none", pad=1),
    )
