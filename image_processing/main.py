import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


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
):
    """
    Executes TOBAC tracking across the specified list of dates and weather phenomena.
    """
    os.makedirs(output_dir, exist_ok=True)
    border_img = cv2.imread(border_img_path, cv2.IMREAD_UNCHANGED)
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(
                _run_tobac_single_day, date, input_dir, output_dir, region, border_img
            ): date
            for date in dates
        }

        for future in as_completed(futures):
            date = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"TOBAC failed for {date}", exc_info=True)

    logger.info("TOBAC runs completed.")


def _run_tobac_single_day(
    date: datetime, input_dir: str, output_dir: str, region: Region, border_img
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
        border_img,
        WeatherPhenomenonTobacParams.TEMPERATURE,
    )
    hum_tra_df, hum_seg_ds = _run_tobac_single_day_single_phenomenon(
        date,
        day_input_dir,
        day_output_dir,
        region,
        WeatherPhenomenon.HUMIDITY,
        border_img,
        WeatherPhenomenonTobacParams.HUMIDITY,
    )
    cld_tra_df, cld_seg_ds = _run_tobac_single_day_single_phenomenon(
        date,
        day_input_dir,
        day_output_dir,
        region,
        WeatherPhenomenon.CLOUDS,
        border_img,
        WeatherPhenomenonTobacParams.CLOUDS,
    )

    results_tra = pd.concat([temp_tra_df, hum_tra_df, cld_tra_df])
    del temp_tra_df, hum_tra_df, cld_tra_df
    results_seg_ds = xr.merge([temp_seg_ds, hum_seg_ds, cld_seg_ds])
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
    border_img,
    phenomenon_params: Optional[WeatherPhenomenonTobacParams] = None,
):
    """
    Runs the TOBAC tracking and visualization pipeline for a single day and phenomenon.
    """
    trajectories_list = []
    segmentations_list = []

    logger.info(f"Processing {phenomenon.value} for {date.strftime('%Y-%m-%d')}")

    for suffix in FOLDERS_HEIGHT_SUFF:
        height_input_dir = os.path.join(day_input_dir, f"{phenomenon.value}{suffix}")
        height_output_dir = os.path.join(day_output_dir, f"{phenomenon.value}{suffix}")
        os.makedirs(height_output_dir, exist_ok=True)

        if not os.path.exists(height_input_dir):
            continue

        image_files = [
            os.path.join(height_input_dir, f)
            for f in os.listdir(height_input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            continue

        image_files = sorted(image_files, key=extract_keys)
        frames = load_image_frames(image_files)
        datetimes = extract_times(image_files)

        is_temp = phenomenon == WeatherPhenomenon.TEMPERATURE
        frames_gray = convert_frames_to_grayscale(frames, is_temperature=is_temp)

        data = np.stack(frames_gray)
        _, frame_height, frame_width = data.shape

        referenced_data = build_referenced_data(
            data, datetimes, region_bounds=region.value
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
            da = xr.concat(x, dim="time")
            da = da.rename(f"{phenomenon.value}{suffix}")
            segmentations_list.append(da)
            del x

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

        # Plotting on original images
        images_no = len(image_files)

        if trajectories is not None and not trajectories.empty:
            trajectories_by_cell = {
                cell: df for cell, df in trajectories.groupby("cell")
            }
        else:
            trajectories_by_cell = {}

        cell_info_by_frame = classify_cells_per_frame(trajectories, DEFAULT_GAP_FRAMES)

        for i in range(images_no):
            original_img_name = os.path.splitext(os.path.basename(image_files[i]))[0]
            out_path = os.path.join(
                height_output_dir, f"{original_img_name}_tracked.png"
            )
            original_img = frames[i]
            cmap = phenomenon_params.value.get("cmap", "viridis")

            generate_plots(
                original_img,
                out_path,
                smooth,
                region,
                segments_all,
                trajectories_by_cell,
                frame_width,
                frame_height,
                cmap,
                cell_info_by_frame,
                border_img,
                i,
            )
        del trajectories
        del segments_all

    if segmentations_list:
        segmentation_ds = xr.merge(segmentations_list)
        del segmentations_list
    else:
        segmentation_ds = xr.Dataset()

    if trajectories_list:
        trajectories_df = pd.concat(trajectories_list, ignore_index=True)
        del trajectories_list
    else:
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

    smoothed_bg = cv2.GaussianBlur(original_img, (0, 0), sigmaX=smooth, sigmaY=smooth)

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


def write_JSON(date, day_output_dir, JSON_clouds, JSON_hum, JSON_temp):
    JSON_of_the_day = {
        "date": date.strftime("%Y-%m-%d"),
        "clouds": JSON_clouds,
        "humidity": JSON_hum,
        "temperature": JSON_temp,
    }
    json_path = os.path.join(
        day_output_dir, date.strftime("%Y-%m-%d") + "_extracted.JSON"
    )
    with open(json_path, "w") as f:
        json.dump(JSON_of_the_day, f, indent=2)
