import logging
import os
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

from image_processing.constants import (
    DEFAULT_GAP_FRAMES,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_SMOOTH,
    DEFAULT_V_MAX,
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
)

logger = logging.getLogger(__name__)


def run_tobac(dates: List[datetime], input_dir: str, output_dir: str, region: Region):
    """
    Executes TOBAC tracking across the specified list of dates and weather phenomena.
    """
    os.makedirs(output_dir, exist_ok=True)

<<<<<<< HEAD
    with ProcessPoolExecutor(max_workers=12) as executor:

        futures = {
            executor.submit(_run_tobac_single_day, date, input_dir, output_dir, region): date
            for date in dates
        }

        for future in as_completed(futures):
            date = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"TOBAC failed for {date}", exc_info=True)
    
    logger.info("TOBAC runs completed.")
=======
    for date in tqdm(dates, desc="Image Processing"):
        day_input_dir = os.path.join(input_dir, date.strftime("%Y-%m-%d"))
        day_output_dir = os.path.join(output_dir, date.strftime("%Y-%m-%d"))
        os.makedirs(day_output_dir, exist_ok=True)
        run_tobac_single_day(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.TEMPERATURE, WeatherPhenomenonTobacParams.TEMPERATURE)
        run_tobac_single_day(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.HUMIDITY, WeatherPhenomenonTobacParams.HUMIDITY)
        run_tobac_single_day(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.CLOUDS, WeatherPhenomenonTobacParams.CLOUDS)
>>>>>>> 52f1f2df8a6b7b89ed0096a79947c29cb2a722ba



def _run_tobac_single_day(
        date: datetime,
        input_dir: str,
        output_dir: str,
        region: Region
):

    day_input_dir = os.path.join(input_dir, date.strftime("%Y-%m-%d"))
    day_output_dir = os.path.join(output_dir, date.strftime("%Y-%m-%d"))
    os.makedirs(day_output_dir, exist_ok=True)
    _run_tobac_single_day_single_phenomenon(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.TEMPERATURE, WeatherPhenomenonTobacParams.TEMPERATURE)
    _run_tobac_single_day_single_phenomenon(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.HUMIDITY, WeatherPhenomenonTobacParams.HUMIDITY)
    _run_tobac_single_day_single_phenomenon(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.CLOUDS, WeatherPhenomenonTobacParams.CLOUDS)




def _run_tobac_single_day_single_phenomenon(
        date: datetime,
        day_input_dir: str,
        day_output_dir: str,
        region: Region,
        phenomenon: WeatherPhenomenon,
        phenomenon_params: Optional[WeatherPhenomenonTobacParams] = None,
):
    """
    Runs the TOBAC tracking and visualization pipeline for a single day and phenomenon.
    """
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

        referenced_data = build_referenced_data(data, datetimes, region_bounds=region.value)
        dxy, dt = get_grid_spacings(referenced_data)
        referenced_data_norm = normalize_referenced_data(referenced_data)

        if phenomenon_params is not None:
            detection_params = phenomenon_params.value
        else:
            detection_params = WeatherPhenomenonTobacParams[phenomenon.name].value

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

        trajectories = track_features(
            features_weighted_points,
            referenced_data,
            dt=dt,
            dxy=dxy,
            v_max=DEFAULT_V_MAX,
            memory=DEFAULT_GAP_FRAMES,
        )

        # Segmentation
        segments_all, all_segment_labels = segment_features(
            features,
            referenced_data_norm,
            threshold=threshold,
            target=target,
            smooth=smooth,
            dxy=dxy,
        )

        # Plotting on original images
        images_no = len(image_files)
        if trajectories is not None and not trajectories.empty:
            trajectories_by_frame = {frame: df for frame, df in trajectories.groupby("frame")}
            trajectories_by_cell = {cell: df for cell, df in trajectories.groupby("cell")}
        else:
            trajectories_by_frame = {}
            trajectories_by_cell = {}

        cells_frames_before = []

        for i in range(images_no):
            itime = i
            original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]
            frame_traj = trajectories_by_frame.get(itime, pd.DataFrame())
            cell_ids = set(frame_traj["cell"].dropna().unique()) if not frame_traj.empty else set()

            all_cells_in_gap = set()
            all_frames_for_cell = {}
            for j in range(DEFAULT_GAP_FRAMES + 1):
                if i - j - 1 >= 0:
                    all_cells_in_gap = all_cells_in_gap | cells_frames_before[i - j - 1]
                    for el in cells_frames_before[i - j - 1]:
                        if el not in all_frames_for_cell:
                            all_frames_for_cell[el] = []
                        all_frames_for_cell[el].append(i - j - 1)

            persisted = cell_ids & all_cells_in_gap
            new_cells = cell_ids - all_cells_in_gap

            # Setup figure
            fig_width_in = frame_width / 100
            fig_height_in = frame_height / 100
            fig, axs = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=100)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Background smoothed image
            original_img = frames[itime]
            smoothed_bg = cv2.GaussianBlur(original_img, (0, 0), sigmaX=smooth, sigmaY=smooth)
            axs.imshow(smoothed_bg, origin="upper")
            xlim = (0, frame_width)
            ylim = (0, frame_height)

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

            entry = next((s for s in segments_all if s[0] == itime), None)
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

            out_path = os.path.join(height_output_dir, f"{original_img_name}_tracked.png")
            plt.savefig(out_path, dpi=100, bbox_inches=None, pad_inches=0)
            plt.close(fig)

            cells_frames_before.append(cell_ids)
