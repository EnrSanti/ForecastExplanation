import numpy as np
import pandas as pd
import xarray as xr

from utils import haversine


def _safe_nanmean(arr: np.ndarray) -> float:
    """nanmean that returns NaN instead of warning on an all-NaN/empty slice."""
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def detect_phenomenon(
    data: xr.Dataset,
    city_masks: dict[str, np.ndarray],
    output_path: str,
    heights: list[str],
    phenomenon: str,
) -> None:
    """
    writes a txt table with:
    timestamp, height, city, {phenomenon}

    the value is the mean over the city's mask (its radius)
    """
    timestamps = pd.to_datetime(data.time.values).strftime("%Y-%m-%d %H:%M:%S")
    records = []

    arrays = {
        h: data[f"raw_{phenomenon}_at_{h}"].transpose("time", ...).values
        for h in heights
    }

    for city_name, mask in city_masks.items():
        for h in heights:
            values = np.nanmean(arrays[h][:, mask], axis=1, dtype=np.float64)
            for timestamp, value in zip(timestamps, values):
                records.append(
                    {
                        "timestamp": timestamp,
                        "height": h.replace("m", ""),
                        "city": city_name,
                        phenomenon: value,
                    }
                )

    columns = ["timestamp", "height", "city", phenomenon]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")


def detect_phenomenon_fronts(
    seg_data: xr.Dataset,
    feat_data: xr.Dataset,
    cities: list[tuple[str, float | int, float | int]],
    output_path: str,
    heights: list[str],
    phenomenon: str,
) -> None:
    """
    writes a txt table with:
    timestamp, height, front_id (from tobac), front area,
     list of cities inside the area, average {phenomenon} inside the
     front, average {phenomenon} outside all fronts (background only)
    """
    dxy_m = float(feat_data.attrs["dxy"])
    area_per_pixel_km2 = (dxy_m / 1000.0) ** 2

    lats = seg_data.latitude.values
    lons = seg_data.longitude.values

    city_pixels = {}
    for city in cities:
        city_name, city_lat, city_lon = city
        dist = haversine(city_lat, city_lon, lats, lons)
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        city_pixels[city_name] = min_idx

    timestamps = pd.to_datetime(seg_data.time.values).strftime("%Y-%m-%d %H:%M:%S")
    records = []

    for h in heights:
        seg_arr = seg_data[f"{phenomenon}_at_{h}"].transpose("time", ...).values
        raw_arr = (
            feat_data[f"raw_{phenomenon}_at_{h}"]
            .sel(time=seg_data.time)
            .transpose("time", ...)
            .values
        )

        for t_idx, timestamp in enumerate(timestamps):
            seg_frame = seg_arr[t_idx]
            val_frame = raw_arr[t_idx]

            # Find unique front IDs (excluding 0, which is background, and nans)
            front_ids = np.unique(seg_frame[~np.isnan(seg_frame)])
            front_ids = [fid for fid in front_ids if fid > 0]

            # Background = not in any front (id 0 or nan), computed once per frame
            background_mask = (seg_frame == 0) | np.isnan(seg_frame)
            avg_val_outside = _safe_nanmean(val_frame[background_mask])

            for fid in front_ids:
                mask = seg_frame == fid

                # Area in pixels
                pixel_count = np.sum(mask)
                area_km2 = pixel_count * area_per_pixel_km2

                # Average value inside this front
                avg_val_inside = _safe_nanmean(val_frame[mask])

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
                        f"{phenomenon}_inside": avg_val_inside,
                        f"{phenomenon}_outside": avg_val_outside,
                    }
                )

    columns = ["timestamp", "height", "front_id", "area", "cities"]
    columns += [f"{phenomenon}_inside", f"{phenomenon}_outside"]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")


def detect_threshold(
    data: xr.Dataset,
    city_masks: dict[str, np.ndarray],
    output_path: str,
    field: str,
    thresholds: dict[str, float],
    sign: int,
) -> None:
    """
    writes a txt table with:
    timestamp, height, city, mean raw_{field} over the city's mask,
    past (city mean beyond the threshold), area (km2 of the region beyond it)

    beyond = sign * value > threshold of the height
    """
    area_per_pixel_km2 = (float(data.attrs["dxy"]) / 1000.0) ** 2
    timestamps = pd.to_datetime(data.time.values).strftime("%Y-%m-%d %H:%M:%S")

    records = []
    for h, threshold in thresholds.items():
        arr = data[f"raw_{field}_at_{h}"].transpose("time", ...).values
        areas = (sign * arr > threshold).sum(axis=(1, 2)) * area_per_pixel_km2
        for name, mask in city_masks.items():
            values = arr[:, mask].mean(axis=1)
            for t_idx, timestamp in enumerate(timestamps):
                records.append(
                    {
                        "timestamp": timestamp,
                        "height": h.replace("m", ""),
                        "city": name,
                        field: values[t_idx],
                        "past": int(sign * values[t_idx] > threshold),
                        "area": int(areas[t_idx]),
                    }
                )

    columns = ["timestamp", "height", "city", field, "past", "area"]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")
