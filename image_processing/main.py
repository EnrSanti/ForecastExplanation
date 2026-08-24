import os
import re
from datetime import datetime
from typing import List

import imageio
import numpy as np
import pandas as pd
import tobac
import xarray as xr

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

        # dxy=5000
        # dt=3600
        # print(referenced_data)

        # normalize all data in the different plots so we can use a single scale/legend and threshold

        vmin = float(referenced_data.min())
        vmax = float(referenced_data.max())
        referenced_data_norm = (referenced_data - vmin) / (vmax - vmin)

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

        pass


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
