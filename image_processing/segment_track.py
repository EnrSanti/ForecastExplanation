import gc
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import tobac
import xarray as xr

from image_processing.constants import (
    DEFAULT_BORDER_THICKNESS,
    DEFAULT_DT,
    DEFAULT_DXY,
    DEFAULT_GAP_FRAMES,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_SMOOTH,
    DEFAULT_V_MAX,
)
from image_processing.split_merge import (
    find_extended_overlap_blobs_inferred,
    get_splits_merges,
)
from image_processing.utils import (
    build_referenced_data,
    convert_frames_to_grayscale,
    extract_keys,
    extract_times,
    get_grid_spacings,
    load_image_frames,
    normalize_referenced_data,
    overlay_image,
)

logger = logging.getLogger(__name__)
logging.getLogger("trackpy").setLevel(logging.WARNING)

DEBUG = False  # Set to True to display search radius circles


def get_blob_positions(trajectories: pd.DataFrame, itime: int) -> str:
    """Returns a string describing movement vectors for cells between itime - 1 and itime."""
    itime_prev = itime - 1
    prev_frames_data = trajectories[trajectories["frame"] == itime_prev]
    curr_frames_data = trajectories[trajectories["frame"] == itime]
    if prev_frames_data.empty or curr_frames_data.empty:
        return ""

    prev_by_cell = prev_frames_data.set_index("cell")
    curr_by_cell = curr_frames_data.set_index("cell")
    common_cells = set(prev_by_cell.index.dropna()).intersection(curr_by_cell.index.dropna())

    all_moved_cells = ""
    for cell_id in sorted(common_cells):
        prev_row = prev_by_cell.loc[cell_id]
        curr_row = curr_by_cell.loc[cell_id]
        prev_x = prev_row["x"].iloc[0] if isinstance(prev_row, pd.DataFrame) else prev_row["x"]
        prev_y = prev_row["y"].iloc[0] if isinstance(prev_row, pd.DataFrame) else prev_row["y"]
        curr_x = curr_row["x"].iloc[0] if isinstance(curr_row, pd.DataFrame) else curr_row["x"]
        curr_y = curr_row["y"].iloc[0] if isinstance(curr_row, pd.DataFrame) else curr_row["y"]
        all_moved_cells += f"Frame {itime}, cell {cell_id} moved from (x: {prev_x}, y:{prev_y} ) to (x: {curr_x}, y:{curr_y} )\n"

    return all_moved_cells


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


import warnings

def track_features(
        features_weighted_points: pd.DataFrame,
        referenced_data: xr.DataArray,
        dt: float = DEFAULT_DT,
        dxy: float = DEFAULT_DXY,
        v_max: float = DEFAULT_V_MAX,
        memory: int = DEFAULT_GAP_FRAMES,
        method_linking: str = "predict",
) -> pd.DataFrame:
    """Links detected features into trajectories across time frames."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Could not generate velocity field for prediction: no tracks"
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
) -> Tuple[List[Tuple[int, Optional[xr.DataArray], Optional[xr.DataArray]]], List[Optional[xr.DataArray]]]:
    """Performs 2D segmentation for each frame in data_norm."""
    segments_all = []
    all_segment_labels = []
    images_no = len(data_norm.time)

    for itime in range(images_no):
        smoothed_frame = ndimage.gaussian_filter(data_norm.isel(time=itime).values, sigma=smooth)
        temp_da = data_norm.isel(time=[itime]).copy()
        temp_da.data = smoothed_frame[np.newaxis, ...]

        f = features[features["frame"] == itime] if features is not None else pd.DataFrame()
        if f.empty:
            segments_all.append((itime, None, None))
            all_segment_labels.append(None)
            continue

        segment_labels, segments = tobac.segmentation_2D(
            f,
            temp_da,
            dxy=dxy,
            threshold=threshold,
            target=target,
        )
        segments_all.append((itime, segment_labels, segments))
        all_segment_labels.append(segment_labels)

    return segments_all, all_segment_labels


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
            alpha = 0.1 + 0.3 * (t0 - track.iloc[0].frame) / (time_diff if time_diff != 0 else 1)
            alpha = max(0.05, min(1.0, alpha))
            axs.plot(line["x"], line["y"], color="blue", linewidth=1.5, alpha=alpha)
    except Exception:
        pass

    if not f_weighted.empty:
        f_weighted.plot.scatter(
            x="x",
            y="y",
            s=40,
            ax=axs,
            color=color,
            marker=printing_symbol,
        )


def print_cloud_labels(f_weighted: pd.DataFrame, cell_id: int, xlim: Tuple[float, float], ylim: Tuple[
    float, float], axs: plt.Axes):
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


def add_circle_slice_filled(ax: plt.Axes, f_weighted: pd.DataFrame, radius: float, xlim: Tuple[float, float], ylim:
Tuple[float, float], color: str = "red", alpha: float = 0.5, **kwargs):
    """Draws search radius polygon around feature."""
    if f_weighted.empty:
        return
    cx = f_weighted["x"].iloc[0]
    cy = f_weighted["y"].iloc[0]

    theta = np.linspace(0, 2 * np.pi, 300)
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    x_clipped = np.clip(x, xlim[0], xlim[1])
    y_clipped = np.clip(y, ylim[0], ylim[1])
    polygon_points = np.column_stack([x_clipped, y_clipped])

    corners = []
    if np.any(x < xlim[0]) and np.any(y < ylim[0]):
        corners.append([xlim[0], ylim[0]])
    if np.any(x > xlim[1]) and np.any(y < ylim[0]):
        corners.append([xlim[1], ylim[0]])
    if np.any(x < xlim[0]) and np.any(y > ylim[1]):
        corners.append([xlim[0], ylim[1]])
    if np.any(x > xlim[1]) and np.any(y > ylim[1]):
        corners.append([xlim[1], ylim[1]])

    if corners:
        polygon_points = np.vstack([polygon_points, corners])

    polygon = patches.Polygon(polygon_points, closed=True, facecolor=color, alpha=alpha, **kwargs)
    ax.add_patch(polygon)

    polygon_border = patches.Polygon(polygon_points, closed=True, facecolor="none", alpha=0.3, edgecolor="red", linestyle="--", linewidth=1, **kwargs)
    ax.add_patch(polygon_border)


def locate_track_merge(
        input_folder: str,
        output_folder: str,
        border_path: Optional[str],
        n_min_threshold: int,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        threshold: float,
        target: str,
        type_: str,
        save_split_merges: bool = True,
        smooth: float = DEFAULT_SMOOTH,
        dxy: float = DEFAULT_DXY,
        dt: float = DEFAULT_DT,
        min_distance: float = DEFAULT_MIN_DISTANCE,
        v_max: float = DEFAULT_V_MAX,
        gap_features_frames: int = DEFAULT_GAP_FRAMES,
):
    """Runs feature detection, tracking, segmentation and split/merge detection."""
    os.makedirs(output_folder, exist_ok=True)

    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                   f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not image_files:
        logger.warning(f"No images found in {input_folder}")
        return

    images_no = len(image_files)
    image_files = sorted(image_files, key=extract_keys)
    frames = load_image_frames(image_files)
    datetimes = extract_times(image_files)

    is_temp = type_ in ["temp", "TEMPERATURE"]
    frames_gray = convert_frames_to_grayscale(frames, is_temperature=is_temp)

    data = np.stack(frames_gray)
    _, n_y, n_x = data.shape

    region_bounds = (lon_min, lon_max, lat_min, lat_max)
    referenced_data = build_referenced_data(data, datetimes, region_bounds=region_bounds)
    calc_dxy, calc_dt = get_grid_spacings(referenced_data, default_dxy=dxy, default_dt=dt)

    test_data_norm = normalize_referenced_data(referenced_data)

    features, features_weighted_points = detect_features(
        test_data_norm,
        threshold=threshold,
        target=target,
        smooth=smooth,
        min_blob_size=n_min_threshold,
        min_distance=min_distance,
        dxy=calc_dxy,
    )

    trajectories = track_features(
        features_weighted_points,
        referenced_data,
        dt=calc_dt,
        dxy=calc_dxy,
        v_max=v_max,
        memory=gap_features_frames,
    )

    segments_all, all_segment_labels = segment_features(
        features,
        test_data_norm,
        threshold=threshold,
        target=target,
        smooth=smooth,
        dxy=calc_dxy,
    )

    cmap = "viridis"
    if type_ in ["cloud", "CLOUDS"]:
        cmap = "viridis"
    elif type_ in ["humidity", "HUMIDITY"]:
        cmap = "YlGnBu"
    elif type_ in ["temp", "TEMPERATURE"]:
        cmap = "OrRd"

    fig_width_in = n_x / 100
    fig_height_in = n_y / 100

    if trajectories is not None and not trajectories.empty:
        trajectories_by_frame = {frame: df for frame, df in trajectories.groupby("frame")}
        trajectories_by_cell = {cell: df for cell, df in trajectories.groupby("cell")}
    else:
        trajectories_by_frame = {}
        trajectories_by_cell = {}

    new_born_at_curr = {}
    disappeared_at_curr = {}
    cells_frames_before: List[Set[int]] = []

    for itime in range(images_no):
        frame_traj = trajectories_by_frame.get(itime, pd.DataFrame())
        cell_ids = set(frame_traj["cell"].dropna().unique()) if not frame_traj.empty else set()

        all_cells_in_gap = set()
        all_frames_for_cell: Dict[int, List[int]] = {}

        for j in range(gap_features_frames + 1):
            if itime - j - 1 >= 0:
                prev_cells = cells_frames_before[itime - j - 1]
                all_cells_in_gap = all_cells_in_gap | prev_cells
                for el in prev_cells:
                    if el not in all_frames_for_cell:
                        all_frames_for_cell[el] = []
                    all_frames_for_cell[el].append(itime - j - 1)

        persisted = cell_ids & all_cells_in_gap
        new_cells = cell_ids - all_cells_in_gap
        disappeared = all_cells_in_gap - cell_ids

        new_born_at_curr[itime] = new_cells
        disappeared_at_curr[itime] = disappeared

        original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]

        fig, axs = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        smoothed_frame = ndimage.gaussian_filter(test_data_norm.isel(time=itime).values, sigma=smooth)
        temp_da = test_data_norm.isel(time=itime).copy()
        temp_da.data = smoothed_frame

        axs.imshow(temp_da.values, origin="upper", cmap=cmap)
        xlim = (0, temp_da.sizes["x"])
        ylim = (0, temp_da.sizes["y"])

        radius = v_max * calc_dt / calc_dxy

        for cell_id in cell_ids:
            track = trajectories_by_cell.get(cell_id, pd.DataFrame())
            f_weighted = track[(track["frame"] == itime)]

            if cell_id in new_cells:
                printing_symbol = "^"
                color = "white"
            else:
                printing_symbol = "x"
                color = "red"

            print_clouds_center_line(printing_symbol, color, f_weighted, itime, track, axs, cell_id, persisted, all_frames_for_cell)

            if len(f_weighted["x"]) > 0:
                print_cloud_labels(f_weighted, cell_id, xlim, ylim, axs)
            if DEBUG:
                add_circle_slice_filled(axs, f_weighted, radius=radius, xlim=xlim, ylim=ylim, color="red", alpha=0.05)

        entry = next((s for s in segments_all if s[0] == itime), None)
        if entry is not None:
            _, seg_labels, _ = entry
            if seg_labels is not None:
                seg_labels2d = seg_labels.isel(time=0)
                seg_labels2d.plot.contour(levels=[0.5], ax=axs, colors="k")

        axs.set_title("")
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_xlim(0, temp_da.sizes["x"])
        axs.set_ylim(temp_da.sizes["y"], 0)
        axs.axis("off")

        overlay_image(border_path, axs, temp_da)

        out_path = os.path.join(output_folder, f"{original_img_name}.png")
        plt.savefig(out_path, dpi=100, bbox_inches=None, pad_inches=0)
        plt.close(fig)

        cells_frames_before.append(cell_ids)

    if save_split_merges:
        blob_positions = ""
        all_splits_merges = ""
        if trajectories is not None and not trajectories.empty:
            for itime in range(1, images_no):
                if all_segment_labels[itime - 1] is None or all_segment_labels[itime] is None:
                    continue

                extended_overlap_map = find_extended_overlap_blobs_inferred(
                    segment_labels=all_segment_labels[itime].isel(time=0).values,
                    trajectories=trajectories,
                    border_thickness_px=DEFAULT_BORDER_THICKNESS,
                )

                splits, merges = get_splits_merges(
                    extended_overlap_map,
                    trajectories,
                    itime,
                    images_no,
                    gap_features_frames,
                    all_segment_labels[itime],
                    all_segment_labels[itime - 1],
                    new_born_at_curr.get(itime, set()),
                    disappeared_at_curr.get(itime, set()),
                )

                blob_positions += get_blob_positions(trajectories, itime)

                if splits != "" or merges != "":
                    all_splits_merges += splits + merges
                    all_splits_merges += "-------------------\n"

            gc.collect()
        with open(os.path.join(output_folder, "split_merge.txt"), "w") as f:
            f.write(str(all_splits_merges))
        with open(os.path.join(output_folder, "movements.txt"), "w") as f:
            f.write(str(blob_positions))

    if trajectories is not None and not trajectories.empty:
        trajectories.to_csv(os.path.join(output_folder, "trajectories.csv"), index=False)
    np.savez_compressed(os.path.join(output_folder, "segment_labels_all.npz"), *all_segment_labels)


def run_tobac_merge_split(
        input_folder: str,
        output_folder: str,
        border_path: Optional[str],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        threshold: float,
        target: str,
        type_: str,
        n_min_threshold: int = 0,
        smooth: float = DEFAULT_SMOOTH,
):
    """Runs feature locate, tracking, segmentation and split/merge detection."""
    locate_track_merge(
        input_folder=input_folder,
        output_folder=output_folder,
        border_path=border_path,
        n_min_threshold=n_min_threshold,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        threshold=threshold,
        target=target,
        type_=type_,
        save_split_merges=True,
        smooth=smooth,
    )
    print("Locating & tracking procedure completed")


def run_tobac_fronts(
        input_folder: str,
        output_folder: str,
        border_path: Optional[str],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        threshold: float,
        target: str,
        type_: str,
        n_min_threshold: int = 0,
        smooth: float = DEFAULT_SMOOTH,
):
    """Runs feature locate and tracking without split/merge detection for fronts."""
    locate_track_merge(
        input_folder=input_folder,
        output_folder=output_folder,
        border_path=border_path,
        n_min_threshold=n_min_threshold,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        threshold=threshold,
        target=target,
        type_=type_,
        save_split_merges=False,
        smooth=smooth,
    )
    print("Locating & tracking (fronts) procedure completed")
