import numpy as np
import pandas as pd
import xarray as xr

from .utils import haversine


def _safe_nanmean(arr: np.ndarray) -> float:
    """nanmean that returns NaN instead of warning on an all-NaN/empty slice."""
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def detect_phenomenon(
    data: xr.Dataset,
    cities: list[tuple[str, float | int, float | int]],
    output_path: str,
    heights: list[str],
    phenomenon: str,
    city_radius: float = 3.0,
) -> None:
    """
    writes a txt table with:
    timestamp, height, lat, lon, {phenomenon}

    the value is the mean of a {city_radius}km radius around the city
    """
    lats = data.latitude.values
    lons = data.longitude.values
    timestamps = pd.to_datetime(data.time.values).strftime("%Y-%m-%d %H:%M:%S")
    records = []

    # Map phenomenon for output column name if needed
    col_name = "temperature" if phenomenon == "temp" else phenomenon

    # Read each variable once instead of once per city/time step
    arrays = {
        v: data[v].transpose("time", ...).values
        for h in heights
        if (v := f"raw_{phenomenon}_at_{h}") in data
    }

    for city in cities:
        _, city_lat, city_lon = city
        dist = haversine(city_lat, city_lon, lats, lons)
        mask = dist <= city_radius

        if not np.any(mask):
            continue

        for h in heights:
            raw_var = f"raw_{phenomenon}_at_{h}"

            if raw_var not in data:
                continue

            for t_idx, timestamp in enumerate(timestamps):
                val_data = arrays[raw_var][t_idx]
                val_mean = _safe_nanmean(val_data[mask])

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

        seg_da = seg_data[seg_var].transpose("time", ...)
        raw_da = feat_data[raw_var].transpose("time", ...)
        seg_arr = seg_da.values
        raw_arr = raw_da.values
        raw_index = {t: i for i, t in enumerate(raw_da.time.values)}
        seg_timestamps = pd.to_datetime(seg_da.time.values).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        for seg_idx, t in enumerate(seg_da.time.values):
            if t not in raw_index:
                continue

            timestamp = seg_timestamps[seg_idx]

            seg_frame = seg_arr[seg_idx]
            val_frame = raw_arr[raw_index[t]]

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
                        f"{col_name}_inside": avg_val_inside,
                        f"{col_name}_outside": avg_val_outside,
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")


def detect_threshold(
    data: xr.Dataset,
    cities: list[tuple[str, float | int, float | int]],
    output_path: str,
    field: str,
    thresholds: dict[str, float],
    sign: int = 1,
    city_radius: float = 3.0,
) -> None:
    """
    writes a txt table with:
    timestamp, height, city, mean raw_{field} in a {city_radius}km radius,
    past (city mean beyond the threshold), area (km2 of the region beyond it)

    beyond = sign * value > threshold of the height
    """
    area_per_pixel_km2 = (float(data.attrs["dxy"]) / 1000.0) ** 2
    lats = data.latitude.values
    lons = data.longitude.values
    timestamps = pd.to_datetime(data.time.values).strftime("%Y-%m-%d %H:%M:%S")
    masks = {
        name: haversine(lat, lon, lats, lons) <= city_radius
        for name, lat, lon in cities
    }

    records = []
    for h, threshold in thresholds.items():
        raw_var = f"raw_{field}_at_{h}"
        if raw_var not in data:
            continue
        arr = data[raw_var].transpose("time", ...).values
        areas = (sign * arr > threshold).sum(axis=(1, 2)) * area_per_pixel_km2
        for name, mask in masks.items():
            if not np.any(mask):
                continue
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

    df = pd.DataFrame(records)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")
