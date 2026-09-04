import logging

import numpy as np
import pandas as pd
import xarray as xr

from .utils import haversine, get_compass_direction

logger = logging.getLogger("ForecastExplanation")


def detect_winds(
    data: xr.Dataset,
    cities: list[tuple[str, float | int, float | int]],
    output_path: str,
    heights: list[str],
    city_radius: float = 3.0,
) -> None:
    """
    writes a txt table with:
    timestamp, height, lat, lon, wind_direction, wind_speed

    lat and lon are of the city
    wind_speed and wind_direction are means of a {city_radius}km radius around the city
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
        mask = dist <= city_radius

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
                        "city": city_name,
                        "wind_direction": get_compass_direction(wd_val),
                        "wind_speed": ws_val,
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")


def detect_clouds(
    seg_data: xr.Dataset,
    feat_data: xr.Dataset,
    cities: list[tuple[str, float | int, float | int]],
    output_path: str,
    heights: list[str],
    city_radius: float = 3.0,
) -> None:
    """
    writes a txt table with:
    timestamp, height, cloud_id, tot area, city, %covered

    tot area is the cloud segment area in km2
    city is the city that is covered
    %covered is the % of the {city_radius}km radius around the city that is covered by the cloud
    """
    if "dxy" not in feat_data.attrs:
        logger.warning("Missing dxy attribute, falling back to 2500m")
        dxy_m = 2500.0
    else:
        dxy_m = float(feat_data.attrs["dxy"])
    area_per_pixel_km2 = (dxy_m / 1000.0) ** 2

    lats = seg_data.latitude.values
    lons = seg_data.longitude.values

    # Pre-calculate city_radius radius mask for each city
    city_masks = {}
    for city in cities:
        city_name, city_lat, city_lon = city
        dist = haversine(city_lat, city_lon, lats, lons)
        mask = dist <= city_radius
        if np.any(mask):
            city_masks[city_name] = mask

    records = []

    for h in heights:
        seg_var = f"cloud_at_{h}"

        if seg_var not in seg_data:
            continue

        for t_idx in range(seg_data.sizes["time"]):
            timestamp = pd.to_datetime(seg_data.time.values[t_idx]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            seg_frame = seg_data[seg_var].isel(time=t_idx).values

            # Find unique cloud IDs (excluding 0, which is background, and nans)
            cloud_ids = np.unique(seg_frame[~np.isnan(seg_frame)])
            cloud_ids = [cid for cid in cloud_ids if cid > 0]

            for cid in cloud_ids:
                cloud_mask = seg_frame == cid

                # Area in pixels
                pixel_count = np.sum(cloud_mask)
                tot_area = pixel_count * area_per_pixel_km2

                # Check intersection with each city's 3km mask
                for city_name, city_mask in city_masks.items():
                    intersection_pixels = np.sum(cloud_mask & city_mask)
                    if intersection_pixels > 0:
                        total_city_pixels = np.sum(city_mask)
                        pct_covered = (intersection_pixels / total_city_pixels) * 100.0

                        records.append(
                            {
                                "timestamp": timestamp,
                                "height": h.replace("m", ""),
                                "cloud_id": int(cid),
                                "tot area": int(tot_area),
                                "city": city_name,
                                "%covered": pct_covered,
                            }
                        )

    df = pd.DataFrame(records)
    if not df.empty:
        # Ensure correct column order
        cols = ["timestamp", "height", "cloud_id", "tot area", "city", "%covered"]
        df = df[cols]
    else:
        df = pd.DataFrame(
            columns=["timestamp", "height", "cloud_id", "tot area", "city", "%covered"]
        )

    df.to_csv(output_path, sep="\t", index=False, float_format="%.2f")
