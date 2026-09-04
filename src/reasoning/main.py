import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from region import Region

logger = logging.getLogger(__name__)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def detect_winds(data: xr.Dataset, cities: list, output_path: str):
    """
    writes a txt table with:
    timestamp, height, lat, lon, wind_direction, wind_speed

    lat and lon are of the city
    wind_speed and wind_direction are means of a 3km radius around the city
    Args:
        data: xarray.Dataset
        cities: list of (name, lat, lon)
    """
    lats = data.latitude.values
    lons = data.longitude.values

    heights = set()
    for var in data.data_vars:
        if "wind_speed_at_" in var:
            heights.add(var.split("wind_speed_at_")[1])

    records = []

    for city in cities:
        city_name, city_lat, city_lon = city
        dist = haversine(city_lat, city_lon, lats, lons)
        mask = dist <= 3.0

        if not np.any(mask):
            continue

        for h in heights:
            ws_var = f"wind_speed_at_{h}"
            wd_var = f"wind_direction_at_{h}"

            for t_idx in range(data.sizes["time"]):
                timestamp = pd.to_datetime(data.time.values[t_idx]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                ws_val = np.nan
                wd_val = np.nan

                if ws_var in data and wd_var in data:
                    ws_data = data[ws_var].isel(time=t_idx).values
                    wd_data = data[wd_var].isel(time=t_idx).values

                    ws_val = np.nanmean(ws_data[mask])

                    wd_rad = np.radians(wd_data[mask])
                    wd_u = np.nanmean(np.sin(wd_rad))
                    wd_v = np.nanmean(np.cos(wd_rad))
                    wd_val = (np.degrees(np.arctan2(wd_u, wd_v)) + 360) % 360

                records.append(
                    {
                        "timestamp": timestamp,
                        "height": h.replace("m", ""),
                        "lat": city_lat,
                        "lon": city_lon,
                        "wind_direction": wd_val,
                        "wind_speed": ws_val,
                    }
                )

    df = pd.DataFrame(records)
    if not df.empty:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, sep="\t", index=False)


def reason(dates: list, input_dir: str, output_dir: str, region: Region, force: bool = False):
    """
    Perform reasoning on the input data (nc format) and save the results to the output path (text format).
    Converts raw data to reasoning data, ready to be converted to ASP formats.

    Args:
        dates (list): List of dates for which reasoning is to be performed.
        input_dir (str): Path to the input directory containing spatial data.
        output_dir (str): Path to the directory where processed output will be written.
        region (Region): The specific geographic region to be used.
        force (bool, optional): If True, forces the processing of all dates. Defaults to False.
    """
    logger.info("Starting reasoning")
    for date in dates:
        day_input_dir = os.path.join(input_dir, date.strftime("%Y-%m-%d"))
        day_output_dir = os.path.join(output_dir, date.strftime("%Y-%m-%d"))
        os.makedirs(day_output_dir, exist_ok=True)

        if not force and os.path.exists(os.path.join(day_output_dir, "reasoning.txt")):
            logger.info(f"Reasoning already exists for {date.strftime('%Y-%m-%d')}. Skipping.")
            continue

        logger.debug(f"Processing reasoning for {date.strftime('%Y-%m-%d')}")

        with xr.open_dataset(os.path.join(day_input_dir, "segmentation.nc")) as ds:
            detect_winds(ds, region.get_cities(), day_output_dir)
