import csv
import io
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from region import Region

from .translator import BaseTranslator

# logger = logging.getLogger("ForecastExplanation")

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
    False: 0,
    "sereno": 0,
    "poco nuvoloso": 1,
    "variabile": 2,
    "nuvoloso": 3,
    "coperto": 4,
    "sole, nebbia, nubi basse": 5,
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

LABEL_COLUMNS = ["prev_pioggia", "prev_cloud"]
DATE_COLUMN = "date"

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


def _read_tsv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.read_text().strip():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, sep="\t")


def _lookup_enum(enum: dict, value, city: str, field: str) -> int:
    if value not in enum:
        raise ValueError(f"Unknown {field} description {value!r} for city {city!r}")
    return enum[value]


def _time_label(timestamp) -> str:
    # return pd.to_datetime(timestamp).strftime("%H%M")
    # this saves around 6 minutes out of 7 for a one year run
    return timestamp[11:13] + timestamp[14:16]


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


def total_cloud_by_time(
    lookup: dict,
    cities: list[str],
    times: list[str],
    heights: list[str],
    round_ndigits: int = 2,
) -> dict:
    """Column cloud cover (hourly max over the levels) per (city, time group),
    plus the whole "day"."""
    buckets: dict[tuple, list] = defaultdict(list)
    for city in cities:
        for t in times:
            vals = [lookup.get((city, t, h)) for h in heights]
            vals = [v for v in vals if v is not None and not pd.isna(v)]
            if not vals:
                continue
            buckets[(city, _hour_group(t))].append(max(vals))
            buckets[(city, "day")].append(max(vals))
    return {k: round(sum(v) / len(v), round_ndigits) for k, v in buckets.items()}


DAILY_COLUMNS = [
    "rh925_max",
    "rh850_max",
    "rh700_max",
    "sat850_hours",
    "sat700_hours",
    "sat_column_hours",
    "south850_mean",
    "south850_max",
    "south700_mean",
    "moist_flux850_mean",
    "moist_flux850_max",
    "lapse_850_500",
    "cloud_low_day",
    "cloud_medium_day",
    "cloud_high_day",
    "cloud_mid_max",
    "cloud_mid_hours",
    "cloud_850_max",
    "te850_mean",
    "te850_drop6h",
    "te850_rise6h",
    "t850_drop6h",
    "instab_te850_500",
]
REGION_COLUMNS = [
    "region_rh850",
    "region_rh700",
    "region_sat700_hours",
    "region_south850",
    "region_cloud_mid",
    "region_te850_tend",
]


def _theta_e(t: float, rh: float, p: float) -> float:
    """Equivalent potential temperature (K) from T (K), RH (%), p (hPa)."""
    tc = t - 273.15
    e = min(max(rh, 1.0), 100.0) / 100 * 6.112 * math.exp(17.67 * tc / (tc + 243.5))
    mixing = 0.622 * e / (p - e)
    return t * (1000.0 / p) ** 0.2854 * math.exp(2.5e6 * mixing / (1004.0 * t))


def _changes_6h(series: dict[int, float]) -> list[float]:
    """6-hour changes of an hourly series (hour 0 = end of the day)."""
    hours = sorted(series, key=lambda h: h or 24)
    by_order = {h or 24: series[h] for h in hours}
    return [by_order[h + 6] - by_order[h] for h in by_order if h + 6 in by_order]


def daily_predictors(
    humidity: dict,
    temperature: dict,
    wind_speed: dict,
    wind_dir: dict,
    cloud_cover: dict,
    cities: list[str],
    round_ndigits: int = 2,
) -> tuple[dict, dict]:
    """Whole-day predictors per city (saturation, southerly moist flow,
    stability, cloud cover, theta-e) and a few pooled over the cities.
    Returns ({city: {column: value}}, {column: value})."""

    def hourly(lookup, city, level):
        out = {}
        for h in range(24):
            v = lookup.get((city, f"{h:02d}00", level))
            if isinstance(v, (int, float)) and not pd.isna(v):
                out[h] = float(v)
        return out

    def south(city, level):
        speed, direction = (
            hourly(wind_speed, city, level),
            hourly(wind_dir, city, level),
        )
        return {
            h: -speed[h] * math.cos(math.radians(direction[h]))
            for h in speed
            if h in direction
        }

    def stat(values, how):
        values = list(values)
        if not values:
            return ""
        return round(how(values), round_ndigits)

    def mean(values):
        return sum(values) / len(values)

    by_city = {}
    pooled = defaultdict(list)
    for city in cities:
        rh = {
            lvl: hourly(humidity, city, lvl) for lvl in ("0925", "0850", "0700", "0500")
        }
        cc = {lvl: hourly(cloud_cover, city, lvl) for lvl in LEVEL_GROUP_MAP}
        s850, s700 = south(city, "0850"), south(city, "0700")
        t850, t500 = (
            hourly(temperature, city, "0850"),
            hourly(temperature, city, "0500"),
        )
        flux = [max(s850[h], 0) * rh["0850"][h] / 100 for h in s850 if h in rh["0850"]]
        te850 = {
            h: _theta_e(t850[h], rh["0850"][h], 850.0) for h in t850 if h in rh["0850"]
        }
        te500 = {
            h: _theta_e(t500[h], rh["0500"][h], 500.0) for h in t500 if h in rh["0500"]
        }
        te_changes = _changes_6h(te850)
        column = [
            h
            for h in rh["0850"]
            if rh["0850"][h] >= 90
            and rh["0925"].get(h, 0) >= 90
            and rh["0700"].get(h, 0) >= 85
        ]
        mid = [
            (cc["0700"][h] + cc["0500"][h]) / 2 for h in cc["0700"] if h in cc["0500"]
        ]
        groups = {
            g: [
                v
                for lvl, g2 in LEVEL_GROUP_MAP.items()
                if g2 == g
                for v in cc[lvl].values()
            ]
            for g in HEIGHT_GROUPS_ORDER
        }
        by_city[city] = {
            "rh925_max": stat(rh["0925"].values(), max),
            "rh850_max": stat(rh["0850"].values(), max),
            "rh700_max": stat(rh["0700"].values(), max),
            "sat850_hours": sum(v >= 90 for v in rh["0850"].values()),
            "sat700_hours": sum(v >= 90 for v in rh["0700"].values()),
            "sat_column_hours": len(column),
            "south850_mean": stat(s850.values(), mean),
            "south850_max": stat(s850.values(), max),
            "south700_mean": stat(s700.values(), mean),
            "moist_flux850_mean": stat(flux, mean),
            "moist_flux850_max": stat(flux, max),
            "lapse_850_500": stat((t850[h] - t500[h] for h in t850 if h in t500), mean),
            "cloud_low_day": stat(groups["low"], mean),
            "cloud_medium_day": stat(groups["medium"], mean),
            "cloud_high_day": stat(groups["high"], mean),
            "cloud_mid_max": stat(mid, max),
            "cloud_mid_hours": sum(v >= 80 for v in mid),
            "cloud_850_max": stat(cc["0850"].values(), max),
            "te850_mean": stat(te850.values(), mean),
            "te850_drop6h": stat((-c for c in te_changes), max),
            "te850_rise6h": stat(te_changes, max),
            "t850_drop6h": stat((-c for c in _changes_6h(t850)), max),
            "instab_te850_500": stat(
                (te850[h] - te500[h] for h in te850 if h in te500), max
            ),
        }
        pooled["rh850"] += rh["0850"].values()
        pooled["rh700"] += rh["0700"].values()
        pooled["s850"] += s850.values()
        pooled["mid"] += mid
        ordered = [te850[h] for h in sorted(te850, key=lambda h: h or 24)]
        if len(ordered) >= 12:
            pooled["te850_tend"].append(mean(ordered[-6:]) - mean(ordered[:6]))

    region = {
        "region_rh850": stat(pooled["rh850"], mean),
        "region_rh700": stat(pooled["rh700"], mean),
        "region_sat700_hours": stat((24 * (v >= 90) for v in pooled["rh700"]), mean),
        "region_south850": stat(pooled["s850"], mean),
        "region_cloud_mid": stat(pooled["mid"], mean),
        "region_te850_tend": stat(pooled["te850_tend"], mean),
    }
    return by_city, region


ROUNDING_STEPS: list[tuple[tuple[str, ...], float | None]] = [
    (
        (
            "size_cloud",
            "region_humidity_fronts",
            "region_temperature_fronts",
            "region_front",
            "region_warmadv",
            "region_tefall",
            "region_terise",
        ),
        100,
    ),  # km2
    (("front_", "warmadv_", "tefall_", "terise_"), 0.1),  # K/100km, K/h, K
    (("te850_mean", "instab"), 1),  # K
    (("te850_", "t850_", "region_te850"), 0.1),  # K
    (("temperature", "lapse"), 1),  # K
    (("cloud_mid_hours",), None),  # a count
    (
        ("humidity", "rh", "region_rh", "cloud", "coverage_clouds", "region_cloud"),
        1,
    ),  # %
    (("wind_speed", "south", "moist_flux", "region_south"), 0.1),  # m/s
]


def _rounding_step(column: str) -> float | None:
    for prefixes, step in ROUNDING_STEPS:
        if column.startswith(prefixes):
            return step
    return None


def round_features(header: list[str], row: list) -> list:
    out = []
    for column, value in zip(header, row):
        step = _rounding_step(column)
        if step is None or isinstance(value, str) or pd.isna(value):
            out.append(value)
        elif step < 1:
            out.append(round(value, 1))
        else:
            out.append(int(round(value / step) * step))
    return out


# reasoning file -> (column prefix, level groups)
FRONT_FILES = {
    "humidity_fronts.txt": ("humidity_fronts", HEIGHT_GROUPS_ORDER),
    "heat_fronts.txt": ("temperature_fronts", HEIGHT_GROUPS_ORDER),
    "front_fronts.txt": ("front", ["low", "medium"]),
    "warmadv_fronts.txt": ("warmadv", ["low", "medium"]),
    "tefall_fronts.txt": ("tefall", ["low", "medium"]),
    "terise_fronts.txt": ("terise", ["low", "medium"]),
}


def _front_columns(prefix: str, groups: list[str]) -> list[str]:
    return [
        *(f"{prefix}_hours_{g}" for g in groups),
        *(f"{prefix}_value_{g}" for g in groups),
        *(f"region_{prefix}_area_{g}" for g in groups),
    ]


def fronts_by_city(
    df: pd.DataFrame, prefix: str, groups: list[str], cities: list[str]
) -> dict:
    """Per city and level group: hours inside a front, mean value inside it,
    and the largest regional front area (km2) of an hour and level."""
    value_col = next((c for c in df.columns if c.endswith("_inside")), None)
    if value_col is None or df.empty:
        df = pd.DataFrame(columns=["timestamp", "height", "area", "cities", "value"])
    else:
        df = df.rename(columns={value_col: "value"})
    group = [LEVEL_GROUP_MAP[_height_label(h)] for h in df["height"]]

    out: dict[str, dict] = {city: {} for city in cities}
    for g in groups:
        sub = df[np.array([x == g for x in group], dtype=bool)]
        areas = sub.groupby(["timestamp", "height"])["area"].sum()
        region_area = int(areas.max()) if len(areas) else 0
        members = [str(c).split(",") for c in sub["cities"]]
        for city in cities:
            inside = sub[np.array([city in m for m in members], dtype=bool)]
            hours = inside["timestamp"].nunique()
            value = inside["value"].mean() if len(inside) else 0.0
            out[city][f"{prefix}_hours_{g}"] = int(hours)
            out[city][f"{prefix}_value_{g}"] = (
                0.0 if pd.isna(value) else round(float(value), 2)
            )
            out[city][f"region_{prefix}_area_{g}"] = region_area
    return out


def _categorical_columns() -> list[str]:
    return ["location"] + [
        col
        for col in _column_group(
            "wind_direction", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER
        )
    ]


class FoldRmTranslator(BaseTranslator):
    extension = "csv"

    @staticmethod
    def schema(header: list[str]) -> dict:
        """Splits a dataset header into label, feature (header order),
        categorical and numeric columns, plus the date column if present."""
        categorical = set(_categorical_columns())
        non_features = set(LABEL_COLUMNS) | {DATE_COLUMN}
        features = [c for c in header if c not in non_features]
        return {
            "labels": [c for c in LABEL_COLUMNS if c in header],
            "date": DATE_COLUMN if DATE_COLUMN in header else None,
            "features": features,
            "categorical": [c for c in features if c in categorical],
            "numeric": [c for c in features if c not in categorical],
        }

    @classmethod
    def schema_from_csv(cls, path: Path) -> dict:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        return cls.schema(header)

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

        cloud_cover_df = _attach_city(
            _read_tsv_or_empty(
                reasoning_dir / "cloud_cover.txt",
                ["timestamp", "height", "lat", "lon", "cloud"],
            ),
            city_coords,
        )

        fronts: dict[str, dict] = {city: {} for city in cities}
        for file_name, (prefix, groups) in FRONT_FILES.items():
            fronts_df = _read_tsv_or_empty(
                reasoning_dir / file_name,
                ["timestamp", "height", "front_id", "area", "cities"],
            )
            for city, columns in fronts_by_city(
                fronts_df, prefix, groups, cities
            ).items():
                fronts[city].update(columns)

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

        cloud_cover, cloud_cover_t, cloud_cover_h = _pivot(
            cloud_cover_df, "cloud", round_ndigits=1
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
        cloud_cover_grouped = average_by_height_and_time(
            cloud_cover, cities, cloud_cover_t, cloud_cover_h
        )
        cloud_total = total_cloud_by_time(
            cloud_cover, cities, cloud_cover_t, cloud_cover_h
        )
        daily, region_daily = daily_predictors(
            humidity, temperature, wind_speed, wind_dir, cloud_cover, cities
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
        header += [
            c for p, groups in FRONT_FILES.values() for c in _front_columns(p, groups)
        ]
        header += _column_group("cloud_cover", TIME_GROUPS_ORDER, HEIGHT_GROUPS_ORDER)
        header += [f"cloud_total_{tg}" for tg in [*TIME_GROUPS_ORDER, "day"]]
        header += DAILY_COLUMNS + REGION_COLUMNS

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
                fronts[city][c]
                for p, groups in FRONT_FILES.values()
                for c in _front_columns(p, groups)
            ]
            row += [
                cloud_cover_grouped.get((city, tg, hg), "")
                for tg in TIME_GROUPS_ORDER
                for hg in HEIGHT_GROUPS_ORDER
            ]
            row += [
                cloud_total.get((city, tg), "") for tg in [*TIME_GROUPS_ORDER, "day"]
            ]
            row += [daily[city][c] for c in DAILY_COLUMNS]
            row += [region_daily[c] for c in REGION_COLUMNS]
            rows.append(round_features(header, row))

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(header)
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")

    def merge_into_dataset(self, dates: list[date], input_folder: Path) -> Path | None:
        if not dates:
            self.logger.error("No dates provided.")
            return None

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
                df.insert(0, DATE_COLUMN, d.isoformat())
                dfs.append(df)
            else:
                self.logger.warning(f"File not found for date {file_path.name}")

        # Merge and save if there's data
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(output_path, index=False)
            self.logger.debug(
                f"Successfully merged {len(dfs)} files into: {output_path}"
            )
            return output_path
        self.logger.error("No matching CSV files found for the given dates.")
        return None
