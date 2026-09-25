import csv
import io
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from region import Region

from .translator import BaseTranslator

_PIOGGIA_ENUM = {
    None: 0,
    False: 0,
    "debole": 6,
    "moderata": 7,
    "abbondante": 8,
    "intensa": 9,
    "molto intensa": 36,
}

_CLOUD_ENUM = {
    None: 0,
    "sereno": 0,
    "poco nuvoloso": 1,
    "variabile": 2,
    "nuvoloso": 3,
    "coperto": 4,
    "sole/nebbia": 5,
}

SUM_CLOUDS = True


# --- Grouping definitions (fixed order, so every day's CSV has the same
# column set regardless of which groups actually had data that day — same
# rationale as the earlier missing-hour backfill). ---
LEVEL_GROUP_MAP = {
    "1000": "low",
    "0925": "low",
    "0850": "low",
    "0700": "medium",
    "0500": "medium",
    "0300": "high",
}

LEVEL_GROUP_COUNT = Counter(LEVEL_GROUP_MAP.values())
HEIGHT_GROUPS_ORDER = ["low", "medium", "high"]
TIME_GROUPS_ORDER = ["early_morning", "morning", "afternoon", "evening"]

CARDINAL_TO_DEG = {
    "N": 0,
    "NE": 45,
    "E": 90,
    "SE": 135,
    "S": 180,
    "SW": 225,
    "W": 270,
    "NW": 315,
}


def _slugify_city(city: str) -> str:
    return city.lower().replace(" ", "_")


def _mean_skip_nan(values: list[float], default: float = 0.0) -> float:
    nums = [v for v in values if not pd.isna(v)]
    return sum(nums) / len(nums) if nums else default


def _front_frequency(detected, total) -> str:
    """Buckets the detected/total frame ratio into "present" (>=0.8),
    "partially_present" (>=0.3), or "absent"."""
    if total == 0:
        return "absent"
    ratio = detected / total
    if ratio >= 0.8:
        return "present"
    elif ratio >= 0.3:
        return "partially_present"
    else:
        return "absent"


def _read_tsv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.read_text().strip():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, sep="\t")


def _lookup_enum(enum: dict, value, city: str, field: str) -> int:
    if value not in enum:
        raise ValueError(f"Unknown {field} description {value!r} for city {city!r}")
    return enum[value]


def _time_label(timestamp) -> str:
    return pd.to_datetime(timestamp).strftime("%H%M")


def _height_label(height) -> str:
    return f"{int(height):04d}"


def _hour_group(time_label: str) -> str:
    """'0000'-'0600' -> early_morning, '0700'-'1200' -> morning,
    '1300'-'1800' -> afternoon, '1900'-'2300' -> evening."""
    hour = int(time_label[:2])
    if hour <= 6:
        return "early_morning"
    elif hour <= 12:
        return "morning"
    elif hour <= 18:
        return "afternoon"
    else:
        return "evening"


def _attach_city(
    df: pd.DataFrame, cities: list[tuple[str, float, float]]
) -> pd.DataFrame:
    """Joins a lat/lon-keyed frame (heat/humidity) to a city name via the
    run's resolved city list (region.get_cities())."""
    latlon_to_city = {(round(lat, 6), round(lon, 6)): city for city, lat, lon in cities}
    df = df.copy()
    df["city"] = [
        latlon_to_city.get((round(lat, 6), round(lon, 6)))
        for lat, lon in zip(df["lat"], df["lon"])
    ]
    return df.dropna(subset=["city"])


def _pivot_no_city(
    df: pd.DataFrame, value_col: list[str], round_ndigits: int | None = None
) -> tuple[dict, list[str], list[str]]:
    times = sorted({_time_label(ts) for ts in df["timestamp"]})
    heights = sorted({_height_label(h) for h in df["height"]})
    keys = zip(
        df["timestamp"].map(_time_label),
        df["height"].map(_height_label),
    )
    found_values = (
        df[value_col].round(round_ndigits).apply(tuple, axis=1)
        if round_ndigits is not None
        else df[value_col].apply(tuple, axis=1)
    )

    lookup = defaultdict(list)
    for key, value in zip(keys, found_values):
        lookup[key].append(value)

    return dict(lookup), times, heights


def _pivot(
    df: pd.DataFrame, value_col: str, round_ndigits: int | None = None
) -> tuple[dict, list[str], list[str]]:
    times = sorted({_time_label(ts) for ts in df["timestamp"]})
    heights = sorted({_height_label(h) for h in df["height"]})
    keys = zip(
        df["city"],
        df["timestamp"].map(_time_label),
        df["height"].map(_height_label),
    )
    values = df[value_col]
    if round_ndigits is not None:
        values = values.round(round_ndigits)
    lookup = dict(zip(keys, values))
    return lookup, times, heights


def _column_group(prefix: str, times: list[str], heights: list[str]) -> list[str]:
    return [f"{prefix}_{t}_{h}" for t in times for h in heights]


def get_compass_direction(degrees: float) -> str | float:
    """Convert a bearing in degrees to an 8-point compass label."""
    if np.isnan(degrees):
        return np.nan
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((degrees + 22.5) // 45) % 8
    return directions[idx]


def average_by_height_and_time(
    lookup: dict,
    cities: list[str],
    times: list[str],
    heights: list[str],
    round_ndigits: int = 2,
) -> dict:
    """
    Groups a (city, time, height)-keyed lookup into (city, time_group,
    height_group), averaging within each group with an arithmetic mean
    that skips missing/non-numeric entries.

    Always groups into the fixed TIME_GROUPS_ORDER / HEIGHT_GROUPS_ORDER —
    a group with no underlying data for a given city just won't have an
    entry in the returned dict, exactly like the existing .get(key, default)
    pattern used when building rows below.
    """
    buckets: dict[tuple, list] = {}
    for city in cities:
        for t in times:
            time_group = _hour_group(t)
            for h in heights:
                height_group = LEVEL_GROUP_MAP.get(h, h)
                val = lookup.get((city, t, h))
                if val is None or val == "":
                    continue
                buckets.setdefault((city, time_group, height_group), []).append(val)

    result = {}
    for key, vals in buckets.items():
        nums = [v for v in vals if isinstance(v, (int, float)) and not pd.isna(v)]
        result[key] = round(sum(nums) / len(nums), round_ndigits) if nums else None

    return result


def average_by_height_and_time_vector(
    speed_lookup: dict,
    dir_lookup: dict,
    cities: list[str],
    times: list[str],
    heights: list[str],
    round_ndigits: int = 2,
) -> dict:
    """
    Groups (city, time, height)-keyed speed+direction lookups into
    (city, time_group, height_group) and computes the resultant wind
    vector for each group: decomposes each (speed, direction) pair into
    u/v components, averages those components within the group, then
    derives resultant speed (vector magnitude) and resultant direction
    (vector bearing, snapped to an 8-point compass label) from that same
    averaged vector — not two independent scalar/circular means.

    Returns a dict keyed the same way as average_by_height_and_time,
    with each value a dict: {"speed": float, "direction": str} (or
    None if there's no valid data for that group).
    """
    buckets: dict[tuple, list] = {}
    for city in cities:
        for t in times:
            time_group = _hour_group(t)
            for h in heights:
                height_group = LEVEL_GROUP_MAP.get(h, h)
                key = (city, t, h)
                speed = speed_lookup.get(key)
                direction = dir_lookup.get(key)
                if (
                    speed is None
                    or speed == ""
                    or direction is None
                    or direction == ""
                    or not isinstance(speed, (int, float))
                    or pd.isna(speed)
                    or not isinstance(direction, (int, float))
                    or pd.isna(direction)
                ):
                    continue
                buckets.setdefault((city, time_group, height_group), []).append(
                    (speed, direction)
                )

    result = {}
    for key, vals in buckets.items():
        u = [speed * math.sin(math.radians(d)) for speed, d in vals]
        v = [speed * math.cos(math.radians(d)) for speed, d in vals]
        if not u:
            result[key] = None
            continue

        u_mean = sum(u) / len(u)
        v_mean = sum(v) / len(v)

        resultant_speed = round(math.hypot(u_mean, v_mean), round_ndigits)
        resultant_deg = math.degrees(math.atan2(u_mean, v_mean)) % 360
        resultant_dir = get_compass_direction(resultant_deg)

        result[key] = {"speed": resultant_speed, "direction": resultant_dir}

    return result


def _get_frames_number(t, h):
    """
    Returns the number of frames (hours) in a given time and height group.
    """
    layer_nos = LEVEL_GROUP_COUNT[h]
    if t == "early_morning":
        return 7 * layer_nos
    elif t == "morning" or t == "afternoon":
        return 6 * layer_nos
    elif t == "evening":
        return 5 * layer_nos
    else:
        raise ValueError(f"Unknown time group: {t}")


def average_by_height_and_time_fronts(
    front_data: dict,
    times: list[str],
    heights: list[str],
    round_ndigits: int = 2,
):
    buckets: dict[tuple, int] = {}
    metric_lists: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)

    for t_all in TIME_GROUPS_ORDER:
        for h_all in HEIGHT_GROUPS_ORDER:
            buckets[(t_all, h_all)] = 0

    for t in times:
        time_group = _hour_group(t)
        for h in heights:
            height_group = LEVEL_GROUP_MAP.get(h, h)
            val = front_data.get((t, h))
            if val is None or val == "":
                continue

            areas = [item[0] for item in val]
            insides = [item[1] for item in val]
            outsides = [item[2] for item in val]

            frame_area = sum(areas)  # sum simultaneous fronts' area
            # avg simultaneous fronts' interior/exterior value, skipping NaN
            # (a front with an empty background at this frame contributes
            # NaN for "outside") so it doesn't wipe out the whole frame
            frame_inside = _mean_skip_nan(insides)
            frame_outside = _mean_skip_nan(outsides)

            buckets[(time_group, height_group)] += 1
            metric_lists[(time_group, height_group)].append(
                (frame_area, frame_inside, frame_outside)
            )

    # front frequency, avg size and avg tmp/hum
    res: dict[tuple, tuple[str, int, float, float]] = {}
    for (t, h), detected in buckets.items():
        vals = metric_lists.get((t, h), [])
        avg_area, avg_inside, avg_outside = 0, 0.0, 0.0
        if vals:
            avg_area = int(sum(v[0] for v in vals) / len(vals))
            avg_inside = round(
                _mean_skip_nan([v[1] for v in vals]), round_ndigits
            )
            avg_outside = round(
                _mean_skip_nan([v[2] for v in vals]), round_ndigits
            )

        res[(t, h)] = (
            _front_frequency(detected, _get_frames_number(t, h)),
            avg_area,
            avg_inside,
            avg_outside,
        )
    return res


class FoldRmTranslator(BaseTranslator):
    extension = "csv"

    def translate_day(
        self, day_input_folder: Path, target_date: date, region: Region
    ) -> bytes:
        gt_data = json.loads((day_input_folder / "gt.json").read_text())
        cities_gt = next(iter(gt_data.values()))
        cities = sorted(cities_gt)
        city_coords = region.get_cities()

        reasoning_dir = day_input_folder / "reasoning"
        winds_df = _read_tsv_or_empty(
            reasoning_dir / "winds.txt",
            ["timestamp", "height", "city", "wind_direction", "wind_speed"],
        )
        cloud_df = pd.read_csv(reasoning_dir / "cloud.txt", sep="\t")
        heat_df = _attach_city(
            _read_tsv_or_empty(
                reasoning_dir / "heat.txt",
                ["timestamp", "height", "lat", "lon", "temperature"],
            ),
            city_coords,
        )
        humidity_df = _attach_city(
            _read_tsv_or_empty(
                reasoning_dir / "humidity.txt",
                ["timestamp", "height", "lat", "lon", "humidity"],
            ),
            city_coords,
        )

        humidity_fronts_df = _read_tsv_or_empty(
            reasoning_dir / "humidity_fronts.txt",
            [
                "timestamp",
                "height",
                "front_id",
                "area",
                "cities",
                "humidity_inside",
                "humidity_outside",
            ],
        )
        heat_fronts_df = _read_tsv_or_empty(
            reasoning_dir / "heat_fronts.txt",
            [
                "timestamp",
                "height",
                "front_id",
                "area",
                "cities",
                "temperature_inside",
                "temperature_outside",
            ],
        )

        if SUM_CLOUDS:
            cloud_df = (
                cloud_df.groupby(["timestamp", "height", "city"]).sum().reset_index()
            )
        else:
            cloud_df = cloud_df.sort_values("%covered").drop_duplicates(
                ["timestamp", "height", "city"], keep="last"
            )

        wind_dir, _, _ = _pivot(winds_df, "wind_direction")
        wind_speed, wind_speed_t, wind_speed_h = _pivot(
            winds_df, "wind_speed", round_ndigits=1
        )

        humidity_front_data, humidity_fronts_t, humidity_fronts_h = _pivot_no_city(
            humidity_fronts_df,
            ["area", "humidity_inside", "humidity_outside"],
            round_ndigits=2,
        )
        temperature_front_data, temperature_fronts_t, temperature_fronts_h = (
            _pivot_no_city(
                heat_fronts_df,
                ["area", "temperature_inside", "temperature_outside"],
                round_ndigits=2,
            )
        )

        coverage, cloud_t, cloud_h = _pivot(cloud_df, "%covered")
        size, _, _ = _pivot(cloud_df, "tot area")

        # cloud_df only has rows for hours where TOBAC actually detected a
        # cloud, so cloud_t can be missing whole hours — and a city with
        # zero detections all day wouldn't appear in cloud_df at all. Force
        # the full 24-hour range and explicitly backfill coverage/size with
        # 0 for every (city, hour, height) combo that wasn't detected, so
        # every day's CSV has the same, stable column set regardless of
        # what TOBAC actually found.
        all_hours = [f"{h:02d}00" for h in range(24)]
        missing_hours = sorted(set(all_hours) - set(cloud_t))

        month = target_date.month

        if missing_hours:
            for city in cities:
                for missing_hour in missing_hours:
                    for h in cloud_h:
                        coverage.setdefault((city, missing_hour, h), 0)
                        size.setdefault((city, missing_hour, h), 0)

        cloud_t = all_hours

        temperature, temp_t, temp_h = _pivot(heat_df, "temperature", round_ndigits=1)
        humidity, humidity_t, humidity_h = _pivot(
            humidity_df, "humidity", round_ndigits=1
        )

        # --- group hours -> early_morning/morning/afternoon/evening and
        # levels -> low/medium/high, averaging within each group ---
        wind_grouped = average_by_height_and_time_vector(
            wind_speed, wind_dir, cities, wind_speed_t, wind_speed_h
        )

        coverage_grouped = average_by_height_and_time(
            coverage, cities, cloud_t, cloud_h
        )
        size_grouped = average_by_height_and_time(size, cities, cloud_t, cloud_h)
        temperature_grouped = average_by_height_and_time(
            temperature, cities, temp_t, temp_h
        )
        humidity_grouped = average_by_height_and_time(
            humidity, cities, humidity_t, humidity_h
        )
        humidity_front_grouped = average_by_height_and_time_fronts(
            humidity_front_data, humidity_fronts_t, humidity_fronts_h
        )
        temp_front_grouped = average_by_height_and_time_fronts(
            temperature_front_data, temperature_fronts_t, temperature_fronts_h
        )
        header = ["prev_pioggia", "prev_cloud", "month", "location"]

        header += _column_group(
            "wind_direction", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER
        )
        header += _column_group("wind_speed", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER)
        header += _column_group(
            "coverage_clouds", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER
        )
        header += _column_group("size_cloud", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER)
        header += _column_group("temperature", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER)
        header += _column_group("humidity", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER)
        header += _column_group(
            "humidity_fronts", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER
        )
        header += _column_group(
            "humidity_fronts_area", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER
        )
        header += _column_group(
            "humidity_fronts_inside_hum", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER
        )
        header += _column_group(
            "humidity_fronts_outside_hum", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER
        )
        header += _column_group(
            "temperature_fronts", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER
        )
        header += _column_group(
            "temperature_fronts_area", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER
        )
        header += _column_group(
            "temperature_fronts_inside_temp", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER
        )
        header += _column_group(
            "temperature_fronts_outside_temp", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER
        )

        rows = []
        for city in cities:
            gt_entry = cities_gt[city]
            pioggia = _lookup_enum(
                _PIOGGIA_ENUM, gt_entry.get("PIOGGIA_DESCRIZIONE"), city, "PIOGGIA"
            )
            cloud = _lookup_enum(
                _CLOUD_ENUM, gt_entry.get("CIELO_DESCRIZIONE"), city, "CIELO"
            )
            slug = _slugify_city(city)
            row = [f"{pioggia}", f"{cloud}", month, slug]

            row += [
                wind_grouped.get((city, tg, hg), {}).get("direction", "")
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                wind_grouped.get((city, tg, hg), {}).get("speed", "")
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                coverage_grouped.get((city, tg, hg), 0)
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                size_grouped.get((city, tg, hg), 0)
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                temperature_grouped.get((city, tg, hg), "")
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                humidity_grouped.get((city, tg, hg), "")
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                humidity_front_grouped.get((tg, hg), "")[0]
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                humidity_front_grouped.get((tg, hg), "")[1]
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                humidity_front_grouped.get((tg, hg), "")[2]
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                humidity_front_grouped.get((tg, hg), "")[3]
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                temp_front_grouped.get((tg, hg), "")[0]
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                temp_front_grouped.get((tg, hg), "")[1]
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                temp_front_grouped.get((tg, hg), "")[2]
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                temp_front_grouped.get((tg, hg), "")[3]
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            rows.append(row)

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(header)
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")

    def merge_into_dataset(self, dates: list[date], input_folder: Path):
        if not dates:
            print("No dates provided.")
            return

        input_path = Path(input_folder)

        # Determine the smallest and biggest dates for the output filename
        min_date = min(dates)
        max_date = max(dates)

        date_format = "%Y-%m-%d"
        ext = self.extension.strip(".") or "csv"
        output_filename = (
            f"{min_date.strftime(date_format)}_{max_date.strftime(date_format)}.{ext}"
        )
        output_path = input_path / output_filename

        # Read and collect dataframes for each date file that exists
        dfs = []
        for d in sorted(dates):
            file_path = input_path / f"{d.strftime(date_format)}.{ext}"
            if file_path.exists():
                df = pd.read_csv(file_path)
                dfs.append(df)
            else:
                print(f"Warning: File not found for date {file_path.name}")

        # Merge and save if there's data
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(output_path, index=False)
            print(f"Successfully merged {len(dfs)} files into: {output_path}")
        else:
            print("No matching CSV files found for the given dates.")
