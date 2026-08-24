import os
import re
from datetime import datetime
from typing import List

import imageio
import numpy as np
import pandas as pd
import tobac
import xarray as xr
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from image_processing.segment_track import print_clouds_center_line, print_cloud_labels, add_circle_slice_filled
from image_processing import Region, WeatherPhenomenon, WeatherPhenomenonTobacPrams, FOLDERS_HEIGHT_SUFF


def run_tobac(dates: List[datetime], input_dir: str, output_dir: str, region: Region):
    # , minumum_size_blob=100, target="upper", save_split_merges=True, smooth=8
    # phenomenon: WeatherPhenomenon, minumum_size_blob, target, save_split_merges=True, smooth=8
    os.makedirs(output_dir, exist_ok=True)

    for date in dates:
        # run sui diversi tipi di fenomeno atmosferico

        day_input_dir = os.path.join(input_dir, date.strftime("%Y-%m-%d"))
        day_output_dir = os.path.join(output_dir, date.strftime("%Y-%m-%d"))
        os.makedirs(day_output_dir, exist_ok=True)
        run_tobac_single_day(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.TEMPERATURE, WeatherPhenomenonTobacPrams.TEMPERATURE)
        run_tobac_single_day(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.HUMIDITY, WeatherPhenomenonTobacPrams.HUMIDITY)
        run_tobac_single_day(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.CLOUDS, WeatherPhenomenonTobacPrams.CLOUDS)
        # run_tobac_single_day(date, input_dir, output_dir, region, WeatherPhenomenon.WIND, WeatherPhenomenonTobacPrams.WIND)


def extract_times(image_files):
    times = []

    for filename in image_files:
        match = re.search(r"_(\d{8})_(\d{4})\.png$", filename)

        if not match:
            raise ValueError(f"Could not extract date from filename: {filename}")

        date_str, time_str = match.groups()
        times.append(pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M") - pd.Timedelta(hours=1))

    return times


def run_tobac_single_day(date: datetime, day_input_dir: str, day_output_dir: str, region: Region, phenomenon: WeatherPhenomenon, pheomenonParams: WeatherPhenomenonTobacPrams):
    # loop only on the possible suffixes of the folders (height)
    for suffix in FOLDERS_HEIGHT_SUFF:
        height_input_dir = os.path.join(day_input_dir, f"{phenomenon.value}{suffix}")
        height_output_dir = os.path.join(day_output_dir, f"{phenomenon.value}{suffix}")
        os.makedirs(height_output_dir, exist_ok=True)
        image_files = ([height_input_dir + "/" + f for f in os.listdir(height_input_dir)
                        if f.lower().endswith(".png")])

        # images_no = len(image_files)

        image_files = sorted(image_files, key=extract_keys)
        frames = [imageio.v2.imread(f) for f in image_files]

        # convert frames to grayscale
        if phenomenon == WeatherPhenomenon.TEMPERATURE:
            frames_gray = [1 - np.mean(f[:, :, :3], axis=2) if f.ndim == 3 else f for f in frames]
        else:
            frames_gray = [np.mean(f[:, :, :3], axis=2) if f.ndim == 3 else f for f in frames]

        # stack into 3D array (time, y, x)
        data = np.stack(frames_gray)
        _, frame_height, frame_width = data.shape

        # set spatial coordinates 
        x_coordinates = np.arange(frame_width)
        y_coordinates = np.arange(frame_height)

        # create xarray.DataArray with the time info
        referenced_data = xr.DataArray(
            data,
            dims=("time", "projection_y_coordinate", "projection_x_coordinate"),
            coords={
                "time": extract_times(image_files),
                "projection_y_coordinate": (
                    "projection_y_coordinate",
                    y_coordinates,
                    {"units": "m"},
                ),
                "projection_x_coordinate": (
                    "projection_x_coordinate",
                    x_coordinates,
                    {"units": "m"},
                ),
            },
            attrs={"units": "m s-1"},
        )

        # set the respective lat/long values to corresp pixels
        lon_min, lon_max, lat_min, lat_max = region.value
        lat = np.linspace(lat_min, lat_max, frame_height)
        lon = np.linspace(lon_min, lon_max, frame_width)
        longitude = np.tile(lon[np.newaxis, :], (frame_height, 1))
        latitude = np.tile(lat[:, np.newaxis], (1, frame_width))

        referenced_data = referenced_data.assign_coords(
            latitude=(("projection_y_coordinate", "projection_x_coordinate"), latitude),
            longitude=(("projection_y_coordinate", "projection_x_coordinate"), longitude))

        # run tobac to get the spacings
        dxy, dt = tobac.get_spacings(referenced_data)

        # normalize all data in the different plots so we can use a single scale/legend and threshold

        vmin = float(referenced_data.min())
        vmax = float(referenced_data.max())
        referenced_data_norm = ((referenced_data - vmin) / (vmax - vmin)).rename({"projection_y_coordinate": "y", "projection_x_coordinate": "x"})

        detection_params = WeatherPhenomenonTobacPrams.TEMPERATURE.value

        min_blob_size = detection_params["min_blob_size"]
        target = detection_params["target"]
        smooth = detection_params["smooth"]
        threshold = detection_params["threshold"]

        # === FEATURE DETECTION ===
        # Locate twice just to get the segmentation right (i.e. with "extreme" i know the center will be inside the object)
        features = tobac.feature_detection_multithreshold(
            referenced_data_norm,
            threshold=[threshold],  # single threshold in normalized space
            dxy=dxy,
            target=target,
            position_threshold="extreme",
            sigma_threshold=smooth,
            n_min_threshold=min_blob_size,
            min_distance=1000  # at least 500m between 2 objects
        )
        # this will be used for getting the center of the objects, the one above for segmentation
        features_weighted_points = tobac.feature_detection_multithreshold(
            referenced_data_norm,
            threshold=[threshold],
            dxy=dxy,
            target=target,
            position_threshold="weighted_abs",
            sigma_threshold=smooth,
            n_min_threshold=min_blob_size,
            min_distance=1000  # at least 1000m between 2 objects
        )

        v_max = 70
        gap_features_frames = 1  # for how many frames a feature can disappear and still be linked (2 full frames in this case, it reappers in the 3)
        radius = v_max * dt / dxy

        # ======== FEATURE TRACKING ========
        try:
            trajectories = tobac.linking_trackpy(features_weighted_points, referenced_data, dt=dt, dxy=dxy, v_max=v_max, memory=gap_features_frames, method_linking="predict")
        except Exception as e:
            print("No trajectories found: ", e)
            trajectories = None

        # ======== SEGMENTING ========
        segments_all = []
        all_segment_labels = []
        images_no = len(image_files)

        for i, itime in enumerate(range(images_no)):
            smoothed_frame = ndimage.gaussian_filter(referenced_data_norm.isel(time=itime).values, sigma=smooth)
            temp_da = referenced_data_norm.isel(time=[itime]).copy()
            temp_da.data = smoothed_frame[np.newaxis, ...]

            f = features[features["frame"] == itime]
            if f.empty:
                segments_all.append((itime, None, None))
                all_segment_labels.append(None)
                continue

            segment_labels, segments = tobac.segmentation_2D(f, temp_da, dxy=dxy, threshold=threshold, target=target)
            segments_all.append((itime, segment_labels, segments))
            all_segment_labels.append(segment_labels)

        # ======== PLOTTING ON ORIGINAL IMAGES ========
        if trajectories is not None:
            cells_frames_before = []
            plot_frames = range(images_no)

            for i, itime in enumerate(plot_frames):
                original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]
                cell_ids = set(trajectories[(trajectories["frame"] == itime)]["cell"].dropna().unique())

                all_cells_in_gap = set()
                all_frames_for_cell = {}
                for j in range(gap_features_frames + 1):
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
                
                # Plot original image
                smoothed_frame = ndimage.gaussian_filter(frames[itime], sigma=smooth)
                axs.imshow(smoothed_frame, origin="upper")
                xlim = (0, frame_width)
                ylim = (0, frame_height)

                for cell_id in cell_ids:
                    track = trajectories[trajectories["cell"] == cell_id]
                    f_weighted = track[(track["frame"] == itime)]

                    if cell_id in new_cells:
                        printing_symbol = '^'
                        color = 'white'
                    else:
                        printing_symbol = 'x'
                        color = 'red'

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
                axs.axis('off')
                
                out_path = os.path.join(height_output_dir, f"{original_img_name}_tracked.png")
                plt.savefig(out_path, dpi=100, bbox_inches=None, pad_inches=0)
                plt.close(fig)
                
                cells_frames_before.append(cell_ids)


def extract_keys(filename):
    """
    extracts date and number from filename for sorting purposes.

    Parameters
    ----------
    filename: file name string

    Returns
    ----------
    tuple (date as int YYYYMMDD, number as int)
    """

    # match pattern like cloud_200_YYYYMMDD_HHMM.png
    m = re.search(r'_(\d{8})_(\d+)\.png$', filename)
    if m:
        date = int(m.group(1))
        num = int(m.group(2))
        return (date, num)
    else:
        return (0, 0)
