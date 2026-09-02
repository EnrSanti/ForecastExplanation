import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional

import cv2
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from image_processing.constants import (
    DEFAULT_GAP_FRAMES,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_SMOOTH,
    DEFAULT_V_MAX_AT_HEIGHT,
    FOLDERS_HEIGHT_SUFF,
    Region,
    WeatherPhenomenon,
    WeatherPhenomenonTobacParams,
)
from image_processing.segment_track import (
    detect_features,
    print_cloud_labels,
    print_clouds_center_line,
    segment_features,
    track_features,
)
from image_processing.utils import (
    build_referenced_data_from_xarray,
    build_referenced_data,
    convert_frames_to_grayscale,
    extract_keys,
    extract_times,
    get_grid_spacings,
    load_image_frames,
    normalize_referenced_data,
    overlay_cities,
)

logger = logging.getLogger(__name__)


def run_tobac(
    dates: List[datetime],
    input_dir: str,
    output_dir: str,
    region: Region,
    border_img_path: str,
    save_images: bool = False,
):
    """
    Executes TOBAC tracking across the specified list of dates and weather phenomena.
    """
    os.makedirs(output_dir, exist_ok=True)
    border_img = cv2.imread(border_img_path, cv2.IMREAD_UNCHANGED)
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(
                _run_tobac_single_day,
                date,
                input_dir,
                output_dir,
                region,
                save_images,
            ): date
            for date in dates
        }

        for future in tqdm(
            as_completed(futures), total=len(dates), desc="TOBAC Processing"
        ):
            date = futures[future]
            try:
                future.result()
            except Exception:
                logger.error(f"TOBAC failed for {date}", exc_info=True)

    logger.info("TOBAC runs completed.")


def _run_tobac_single_day(
    date: datetime,
    input_dir: str,
    output_dir: str,
    region: Region,
    save_images: bool = False,
):
    day_input_dir = os.path.join(input_dir, date.strftime("%Y-%m-%d"))
    day_output_dir = os.path.join(output_dir, date.strftime("%Y-%m-%d"))
    os.makedirs(day_output_dir, exist_ok=True)

    temp_tra_df, temp_seg_ds = _run_tobac_single_day_single_phenomenon(
        date,
        day_input_dir,
        day_output_dir,
        region,
        WeatherPhenomenon.TEMPERATURE,
        WeatherPhenomenonTobacParams.TEMPERATURE,
        save_images,
    )
    hum_tra_df, hum_seg_ds = _run_tobac_single_day_single_phenomenon(
        date,
        day_input_dir,
        day_output_dir,
        region,
        WeatherPhenomenon.HUMIDITY,
        WeatherPhenomenonTobacParams.HUMIDITY,
        save_images,
    )
    cld_tra_df, cld_seg_ds = _run_tobac_single_day_single_phenomenon(
        date,
        day_input_dir,
        day_output_dir,
        region,
        WeatherPhenomenon.CLOUDS,
        WeatherPhenomenonTobacParams.CLOUDS,
        save_images,
    )

    dfs = [df for df in [temp_tra_df, hum_tra_df, cld_tra_df] if not df.empty]
    results_tra = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    del temp_tra_df, hum_tra_df, cld_tra_df
    results_seg_ds = xr.merge(
        [temp_seg_ds, hum_seg_ds, cld_seg_ds], compat="override", join="outer"
    )
    del temp_seg_ds, hum_seg_ds, cld_seg_ds

    logger.debug(f"total space used for day {date.strftime('%Y-%m-%d')}:")
    logger.debug(results_tra.memory_usage())

    xr.Dataset.from_dataframe(results_tra).to_netcdf(
        os.path.join(day_output_dir, "trajectories.nc")
    )
    results_seg_ds.to_netcdf(os.path.join(day_output_dir, "segmentation.nc"))


def _run_tobac_single_day_single_phenomenon(
    date: datetime,
    day_input_dir: str,
    day_output_dir: str,
    region: Region,
    phenomenon: WeatherPhenomenon,
    phenomenon_params: Optional[WeatherPhenomenonTobacParams] = None,
    save_images: bool = False,
):
    """
    Runs the TOBAC tracking and visualization pipeline for a single day and phenomenon.
    """
    trajectories_list = []
    segmentations_list = []

    logger.debug(f"Processing {phenomenon.value} for {date.strftime('%Y-%m-%d')}")

    for suffix in FOLDERS_HEIGHT_SUFF:
        features_nc = os.path.join(day_input_dir, "features.nc")

        if not os.path.exists(features_nc):
            continue

        with xr.open_dataset(features_nc) as ds:
            folder_key = f"{phenomenon.value}{suffix}"
            if folder_key not in ds:
                continue

            da = ds[folder_key].load()

        datetimes = [pd.Timestamp(t) for t in da.time.values]

        referenced_data = build_referenced_data_from_xarray(
            da, datetimes, region_bounds=region.value
        )
        dxy, dt = get_grid_spacings(referenced_data)
        referenced_data_norm = normalize_referenced_data(referenced_data)

        if phenomenon_params is None:
            phenomenon_params = WeatherPhenomenonTobacParams[phenomenon.name]

        detection_params = phenomenon_params.value

        min_blob_size = detection_params.get("min_blob_size", 100)
        target = detection_params.get("target", "maximum")
        smooth = detection_params.get("smooth", DEFAULT_SMOOTH)
        threshold = detection_params.get("threshold", 0.6)

        # Feature detection & tracking
        features, features_weighted_points = detect_features(
            referenced_data_norm,
            threshold=threshold,
            target=target,
            smooth=smooth,
            min_blob_size=min_blob_size,
            min_distance=DEFAULT_MIN_DISTANCE,
            dxy=dxy,
        )
        v_max_at_height = DEFAULT_V_MAX_AT_HEIGHT.get(suffix, 60)
        trajectories = track_features(
            features_weighted_points,
            referenced_data,
            dt=dt,
            dxy=dxy,
            v_max=v_max_at_height,
            memory=DEFAULT_GAP_FRAMES,
        )

        # Segmentation
        segments_all = segment_features(
            features,
            referenced_data_norm,
            threshold=threshold,
            target=target,
            smooth=smooth,
            dxy=dxy,
        )

        if x := [s[1] for s in segments_all if s[1] is not None]:
            seg_da = xr.concat(x, dim="time")
            seg_da = seg_da.rename(f"{phenomenon.value}{suffix}")
            segmentations_list.append(seg_da)
            del x

        if save_images:
            from image_processing.plotting import generate_all_plots

            height_output_dir = os.path.join(day_output_dir, folder_key)
            generate_all_plots(
                da=da,
                output_dir=height_output_dir,
                cmap=detection_params.get("cmap", "viridis"),
                region=region,
                segments_all=segments_all,
                trajectories=trajectories,
            )

        if trajectories is not None and not trajectories.empty:
            tmp = trajectories.drop(
                columns=[
                    "frame",
                    "idx",
                    "threshold_value",
                    "feature",
                    "timestr",
                    "y",
                    "x",
                ],
                errors="ignore",
            )
            tmp["height"] = f"{phenomenon.value}{suffix}"
            tmp["height"] = tmp["height"].astype("category")
            trajectories_list.append(tmp)

        del trajectories
        del segments_all

    if segmentations_list:
        segmentation_ds = xr.merge(segmentations_list, compat="override", join="outer")
        del segmentations_list
    else:
        segmentation_ds = xr.Dataset()

    if trajectories_list:
        valid_dfs = [df for df in trajectories_list if not df.empty]
        if valid_dfs:
            trajectories_df = pd.concat(valid_dfs, ignore_index=True)
        else:
            trajectories_df = None
        del trajectories_list
    else:
        trajectories_df = None

    if trajectories_df is None:
        trajectories_df = pd.DataFrame(
            columns=[
                "hdim_1",
                "hdim_2",
                "num",
                "time",
                "latitude",
                "longitude",
                "cell",
                "time_cell",
                "height",
            ]
        )

    plt.close("all")
    return trajectories_df, segmentation_ds


def classify_cells_per_frame(trajectories: pd.DataFrame, gap_frames: int) -> dict:
    """
    For each frame, classify which cells are present, and split them into
    'persisted' (also seen in the gap_frames window immediately before this
    frame) vs 'new' (first appearance, or reappearing after a longer gap
    than trackpy's memory bridged).

    Returns {frame: {"cell_ids": set, "persisted": set, "new_cells": set,
                      "all_frames_for_cell": {cell_id: [prior_frames_in_window]}}}

    Replaces the manual cells_frames_before + nested gap-window loop: since
    trajectories already carries each cell's full frame history (trackpy's
    `memory` already did the gap-bridging when linking), this just reads
    that history directly per cell instead of replaying it frame-by-frame.
    """
    if trajectories is None or trajectories.empty:
        return {}

    frames_by_cell = trajectories.groupby("cell")["frame"].apply(
        lambda s: sorted(s.unique())
    )

    result = {}
    for i, frame_traj in trajectories.groupby("frame"):
        cell_ids = set(frame_traj["cell"].dropna().unique())
        persisted, new_cells, all_frames_for_cell = set(), set(), {}

        for cell_id in cell_ids:
            cell_frames = frames_by_cell.get(cell_id, [])
            recent = [f for f in cell_frames if i - gap_frames - 1 <= f < i]
            if recent:
                persisted.add(cell_id)
                all_frames_for_cell[cell_id] = recent
            else:
                new_cells.add(cell_id)

        result[i] = {
            "cell_ids": cell_ids,
            "persisted": persisted,
            "new_cells": new_cells,
            "all_frames_for_cell": all_frames_for_cell,
        }
    return result


def generate_plots(
    original_img,
    out_path: str,
    smooth: float,
    region,
    segments_all,
    trajectories_by_cell,
    frame_width,
    frame_height,
    cmap,
    cell_info_by_frame,
    border_img,
    i,
):
    fig_width_in = frame_width / 100
    fig_height_in = frame_height / 100

    xlim = (0, frame_width)
    ylim = (0, frame_height)

    fig, axs = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    smoothed_bg = cv2.GaussianBlur(
        np.asarray(original_img), (0, 0), sigmaX=smooth, sigmaY=smooth
    )

    axs.imshow(smoothed_bg, origin="upper", cmap=cmap)
    axs.imshow(border_img, origin="upper")
    overlay_cities(axs, region, 800, 915)  # TODO: da non hardcodare
    info = cell_info_by_frame.get(
        i,
        {
            "cell_ids": set(),
            "persisted": set(),
            "new_cells": set(),
            "all_frames_for_cell": {},
        },
    )
    cell_ids = info["cell_ids"]
    persisted = info["persisted"]
    new_cells = info["new_cells"]
    all_frames_for_cell = info["all_frames_for_cell"]

    # ... figure setup, imshow, overlay_image, overlay_cities unchanged ...

    for cell_id in cell_ids:
        track = trajectories_by_cell.get(cell_id, pd.DataFrame())
        f_weighted = track[track["frame"] == i]

        printing_symbol, color = (
            ("^", "white") if cell_id in new_cells else ("x", "red")
        )
        print_clouds_center_line(
            printing_symbol,
            color,
            f_weighted,
            i,
            track,
            axs,
            cell_id,
            persisted,
            all_frames_for_cell,
        )

        if len(f_weighted["x"]) > 0:
            print_cloud_labels(f_weighted, cell_id, xlim, ylim, axs)

    entry = next((s for s in segments_all if s[0] == i), None)
    if entry is not None:
        _, seg_labels, _ = entry
        if seg_labels is not None:
            seg_labels2d = seg_labels.isel(time=0)
            seg_labels2d.plot.contour(levels=[0.5], ax=axs, colors="k")

    axs.set_title("")
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_xlim(0, frame_width)
    axs.set_ylim(frame_height, 0)
    axs.axis("off")

    plt.savefig(out_path, dpi=100, bbox_inches=None, pad_inches=0)
    fig.clf()
    plt.close(fig)
