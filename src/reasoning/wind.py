import logging
import pandas as pd
import xarray as xr
import numpy as np

from .utils import haversine, get_compass_direction

logger = logging.getLogger("ForecastExplanation")


def detect_winds(
    data: xr.Dataset,
    cities: list[tuple[str, float | int, float | int]],
    output_path: str,
    heights: list[str],
) -> None:
    """
    writes a txt table with:
    timestamp, height, lat, lon, wind_direction, wind_speed

    lat and lon are of the city
    wind_speed and wind_direction are means of a 3km radius around the city
    Args:
        data: xarray.Dataset
        cities: list of (name, lat, lon)
        heights: list of heights to consider
        output_path: path to the output file
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
            ws_var = f"wind_at_{h}"
            wd_var = f"wind_direction_at_{h}"

            for t_idx in range(data.sizes["time"]):
                timestamp = pd.to_datetime(data.time.values[t_idx]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                if ws_var not in data or wd_var not in data:
                    missing = [v for v in [ws_var, wd_var] if v not in data]
                    logger.warning(
                        f"Skipping {city_name} at height {h}, timestamp {timestamp}: "
                        f"missing variable(s) {missing}"
                    )
                    continue

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
                        "wind_direction": get_compass_direction(wd_val),
                        "wind_speed": ws_val,
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")
