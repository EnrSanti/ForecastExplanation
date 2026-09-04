import xarray as xr
import numpy as np
import pandas as pd

from .utils import haversine

import os


def detect_heat(
    data: xr.Dataset,
    cities: list[tuple[str, float | int, float | int]],
    output_path: str,
    heights: list[str],
) -> None:
    """
    writes a txt table with:
    timestamp, height, lat, lon, temperature

    the temperature is the mean of a 3km radius around the city
    """
    lats = data.latitude.values
    lons = data.longitude.values
    records = []

    for city in cities:
        city_name, city_lat, city_lon = city
        dist = haversine(city_lat, city_lon, lats, lons)
        mask = dist <= 3.0

        if not np.any(mask):
            continue

        for h in heights:
            raw_temp_var = f"raw_temp_at_{h}"

            if raw_temp_var not in data:
                continue

            for t_idx in range(data.sizes["time"]):
                timestamp = pd.to_datetime(data.time.values[t_idx]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                temp_data = data[raw_temp_var].isel(time=t_idx).values
                temp_val = np.nanmean(temp_data[mask])

                records.append(
                    {
                        "timestamp": timestamp,
                        "height": h.replace("m", ""),
                        "lat": city_lat,
                        "lon": city_lon,
                        "temperature": temp_val,
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")
