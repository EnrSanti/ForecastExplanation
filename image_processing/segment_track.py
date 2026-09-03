import logging
import warnings
from typing import List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
import tobac
import xarray as xr

matplotlib.use("Agg")

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
