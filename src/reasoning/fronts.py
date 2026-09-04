import xarray as xr
import numpy as np
import pandas as pd

from .utils import haversine
from .constants import CITY_RADIUS_KM

import os


def detect_phenomenon(
    data: xr.Dataset,
    cities: list[tuple[str, float | int, float | int]],
    output_path: str,
    heights: list[str],
    phenomenon: str,
) -> None:
    """
    writes a txt table with:
    timestamp, height, lat, lon, {phenomenon}

    the value is the mean of a {CITY_RADIUS_KM}km radius around the city
    """
    lats = data.latitude.values
    lons = data.longitude.values
    records = []

    # Map phenomenon for output column name if needed
    col_name = "temperature" if phenomenon == "temp" else phenomenon

    for city in cities:
        city_name, city_lat, city_lon = city
        dist = haversine(city_lat, city_lon, lats, lons)
        mask = dist <= CITY_RADIUS_KM

        if not np.any(mask):
            continue

        for h in heights:
            raw_var = f"raw_{phenomenon}_at_{h}"

            if raw_var not in data:
                continue

            for t_idx in range(data.sizes["time"]):
                timestamp = pd.to_datetime(data.time.values[t_idx]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                val_data = data[raw_var].isel(time=t_idx).values
                val_mean = np.nanmean(val_data[mask])

                records.append(
                    {
                        "timestamp": timestamp,
                        "height": h.replace("m", ""),
                        "lat": city_lat,
                        "lon": city_lon,
                        col_name: val_mean,
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")


def detect_phenomenon_fronts(
    seg_data: xr.Dataset,
    feat_data: xr.Dataset,
    cities: list[tuple[str, float | int, float | int]],
    output_path: str,
    heights: list[str],
    phenomenon: str,
):
    """
    writes a txt table with:
    timestamp, height, front_id (from tobac), front area, list of cities inside the area, average {phenomenon} of the front
    """
    dxy_m = float(feat_data.attrs["dxy"])
    area_per_pixel_km2 = (dxy_m / 1000.0) ** 2

    lats = seg_data.latitude.values
    lons = seg_data.longitude.values

    # Map phenomenon for output column name if needed
    col_name = "temperature" if phenomenon == "temp" else phenomenon

    # Pre-calculate nearest pixel index for each city
    city_pixels = {}
    for city in cities:
        city_name, city_lat, city_lon = city
        dist = haversine(city_lat, city_lon, lats, lons)
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        city_pixels[city_name] = min_idx

    records = []

    for h in heights:
        seg_var = f"{phenomenon}_at_{h}"
        raw_var = f"raw_{phenomenon}_at_{h}"

        if seg_var not in seg_data or raw_var not in feat_data:
            continue

        for t_idx in range(seg_data.sizes["time"]):
            timestamp = pd.to_datetime(seg_data.time.values[t_idx]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            seg_frame = seg_data[seg_var].isel(time=t_idx).values
            val_frame = feat_data[raw_var].isel(time=t_idx).values

            # Find unique front IDs (excluding 0, which is background, and nans)
            front_ids = np.unique(seg_frame[~np.isnan(seg_frame)])
            front_ids = [fid for fid in front_ids if fid > 0]

            for fid in front_ids:
                mask = seg_frame == fid

                # Area in pixels
                pixel_count = np.sum(mask)
                area_km2 = pixel_count * area_per_pixel_km2

                # Average value
                avg_val = np.nanmean(val_frame[mask])

                # Cities inside this front
                cities_inside = []
                for city_name, idx in city_pixels.items():
                    if seg_frame[idx] == fid:
                        cities_inside.append(city_name)

                cities_str = ",".join(cities_inside) if cities_inside else "none"

                records.append(
                    {
                        "timestamp": timestamp,
                        "height": h.replace("m", ""),
                        "front_id": int(fid),
                        "area": int(area_km2),
                        "cities": cities_str,
                        col_name: avg_val,
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")
