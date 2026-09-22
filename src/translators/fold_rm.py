import csv
import io
import json
import math
from pathlib import Path

import pandas as pd

from region import CITIES

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


def _circular_mean_direction(directions: list[str]) -> str | None:
    """
    Circular mean of 8-point cardinal directions, snapped back to the
    nearest cardinal label. Never average the compass angles arithmetically
    (N=0/360 and NE=45 would wrongly average toward NNE-ish nonsense near
    the wrap-around) — this goes through sin/cos averaging instead.

    Note: when directions are perfectly symmetric (e.g. N/E/S/W evenly
    represented), the true resultant vector has ~zero magnitude — there is
    genuinely no dominant direction — but floating-point noise in sin/cos
    can still snap the result to an arbitrary-looking cardinal label rather
    than something meaningful. Rare for real 6-hour wind windows, but worth
    knowing if you ever see a surprising direction for a mixed-direction
    period.
    """
    valid = [d for d in directions if d in CARDINAL_TO_DEG]
    if not valid:
        return None

    sin_sum = sum(math.sin(math.radians(CARDINAL_TO_DEG[d])) for d in valid)
    cos_sum = sum(math.cos(math.radians(CARDINAL_TO_DEG[d])) for d in valid)
    mean_deg = math.degrees(math.atan2(sin_sum, cos_sum)) % 360

    def _angular_dist(a, b):
        return min(abs(a - b), 360 - abs(a - b))

    nearest = min(
        CARDINAL_TO_DEG.items(), key=lambda kv: _angular_dist(kv[1], mean_deg)
    )
    return nearest[0]


def _attach_city(df: pd.DataFrame) -> pd.DataFrame:
    """Joins a lat/lon-keyed frame (heat/humidity) to a city name via region.CITIES."""
    latlon_to_city = {
        (round(info["lat"], 6), round(info["lon"], 6)): city
        for city, info in CITIES.items()
    }
    df = df.copy()
    df["city"] = [
        latlon_to_city.get((round(lat, 6), round(lon, 6)))
        for lat, lon in zip(df["lat"], df["lon"])
    ]
    return df.dropna(subset=["city"])


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


def average_by_height_and_time(
    lookup: dict,
    cities: list[str],
    times: list[str],
    heights: list[str],
    kind: str = "mean",
    round_ndigits: int = 2,
) -> dict:
    """
    Groups a (city, time, height)-keyed lookup into (city, time_group,
    height_group), averaging within each group.

    kind="mean"      -> arithmetic mean, skipping missing/non-numeric entries
    kind="direction" -> circular mean of 8-point cardinal directions (see
                         _circular_mean_direction), snapped back to a label

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
        if kind == "direction":
            result[key] = _circular_mean_direction(vals)
        else:
            nums = [v for v in vals if isinstance(v, (int, float)) and not pd.isna(v)]
            result[key] = round(sum(nums) / len(nums), round_ndigits) if nums else None

    return result


class FoldRmTranslator(BaseTranslator):
    extension = "csv"

    def translate_day(self, day_input_folder: Path) -> bytes:
        gt_data = json.loads((day_input_folder / "gt.json").read_text())
        cities_gt = next(iter(gt_data.values()))
        cities = sorted(cities_gt)

        reasoning_dir = day_input_folder / "reasoning"
        winds_df = pd.read_csv(reasoning_dir / "winds.txt", sep="\t")
        cloud_df = pd.read_csv(reasoning_dir / "cloud.txt", sep="\t")
        heat_df = _attach_city(pd.read_csv(reasoning_dir / "heat.txt", sep="\t"))
        humidity_df = _attach_city(
            pd.read_csv(reasoning_dir / "humidity.txt", sep="\t")
        )

        if SUM_CLOUDS:
            cloud_df = (
                cloud_df.groupby(["timestamp", "height", "city"]).sum().reset_index()
            )
        else:
            cloud_df = cloud_df.sort_values("%covered").drop_duplicates(
                ["timestamp", "height", "city"], keep="last"
            )

        wind_dir, wind_dir_t, wind_dir_h = _pivot(winds_df, "wind_direction")
        wind_speed, wind_speed_t, wind_speed_h = _pivot(
            winds_df, "wind_speed", round_ndigits=1
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
        wind_dir_grouped = average_by_height_and_time(
            wind_dir, cities, wind_dir_t, wind_dir_h, kind="direction"
        )
        wind_speed_grouped = average_by_height_and_time(
            wind_speed, cities, wind_speed_t, wind_speed_h, kind="mean"
        )
        coverage_grouped = average_by_height_and_time(
            coverage, cities, cloud_t, cloud_h, kind="mean"
        )
        size_grouped = average_by_height_and_time(
            size, cities, cloud_t, cloud_h, kind="mean"
        )
        temperature_grouped = average_by_height_and_time(
            temperature, cities, temp_t, temp_h, kind="mean"
        )
        humidity_grouped = average_by_height_and_time(
            humidity, cities, humidity_t, humidity_h, kind="mean"
        )

        header = ["prev_pioggia", "prev_cloud"]
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
            row = [f"{slug}_{pioggia}", f"{slug}_{cloud}"]

            row += [
                wind_dir_grouped.get((city, tg, hg), "")
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                wind_speed_grouped.get((city, tg, hg), "")
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

            rows.append(row)

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(header)
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")
