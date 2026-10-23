from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .utils import pixel_area_km2


def detect_winds(
    data: xr.Dataset,
    city_masks: dict[str, np.ndarray],
    output_path: Path,
    heights: list[str],
) -> None:
    """
    writes a txt table with:
    timestamp, height, city, wind_direction, wind_speed

    wind_speed is the mean and wind_direction the direction of the mean wind
    vector over the city's mask (its radius)
    """
    timestamps = pd.to_datetime(data.time.values).strftime("%Y-%m-%d %H:%M:%S")

    # Read each variable once instead of once per city
    arrays = {
        h: (
            data[f"wind_at_{h}"].transpose("time", ...).values,
            np.radians(data[f"wind_direction_at_{h}"].transpose("time", ...).values),
        )
        for h in heights
    }

    records = []

    for city_name, mask in city_masks.items():
        for h in heights:
            speed, direction = arrays[h][0][:, mask], arrays[h][1][:, mask]
            mean_speed = np.nanmean(speed, axis=1, dtype=np.float64)
            u = np.nanmean(speed * np.sin(direction), axis=1, dtype=np.float64)
            v = np.nanmean(speed * np.cos(direction), axis=1, dtype=np.float64)
            mean_direction = (np.degrees(np.arctan2(u, v)) + 360) % 360

            for timestamp, wd, ws in zip(timestamps, mean_direction, mean_speed):
                records.append(
                    {
                        "timestamp": timestamp,
                        "height": h.replace("m", ""),
                        "city": city_name,
                        "wind_direction": wd,
                        "wind_speed": ws,
                    }
                )

    columns = ["timestamp", "height", "city", "wind_direction", "wind_speed"]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")


def detect_clouds(
    seg_data: xr.Dataset,
    feat_data: xr.Dataset,
    city_masks: dict[str, np.ndarray],
    output_path: Path,
    heights: list[str],
) -> None:
    """
    writes a txt table with:
    timestamp, height, cloud_id, tot area, city, %covered

    tot area is the cloud segment area in km2
    city is the city that is covered
    %covered is the % of the city's mask (its radius) covered by the cloud
    """
    area_per_pixel_km2 = pixel_area_km2(feat_data)

    timestamps = pd.to_datetime(seg_data.time.values).strftime("%Y-%m-%d %H:%M:%S")

    records = []

    for h in heights:
        seg_arr = seg_data[f"cloud_at_{h}"].transpose("time", ...).values

        for t_idx, timestamp in enumerate(timestamps):
            seg_frame = seg_arr[t_idx]

            # cloud ids, 0 and NaN being background
            cloud_ids = np.unique(seg_frame[~np.isnan(seg_frame)])
            cloud_ids = [cid for cid in cloud_ids if cid > 0]

            for cid in cloud_ids:
                cloud_mask = seg_frame == cid

                pixel_count = np.sum(cloud_mask)
                tot_area = pixel_count * area_per_pixel_km2

                for city_name, city_mask in city_masks.items():
                    intersection_pixels = np.sum(cloud_mask & city_mask)
                    if intersection_pixels > 0:
                        total_city_pixels = np.sum(city_mask)
                        pct_covered = round(
                            (intersection_pixels / total_city_pixels) * 100.0
                        )

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

    columns = ["timestamp", "height", "cloud_id", "tot area", "city", "%covered"]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.2f")
