import csv
import io
import itertools
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .translator import BaseTranslator

_PIOGGIA_ENUM = {
    False: 0,
    "debole": 6,
    "moderata": 7,
    "abbondante": 8,
    "intensa": 9,
    "molto intensa": 36,
}

_CLOUD_ENUM = {
    False: 0,
    "sereno": 0,
    "poco nuvoloso": 1,
    "variabile": 2,
    "nuvoloso": 3,
    "coperto": 4,
    "sole, nebbia, nubi basse": 5,
}

# fixed order, so every day's CSV has the same columns
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


def _slugify_city(city: str) -> str:
    return city.lower().replace(" ", "_")


def _lookup_enum(enum: dict, value, city: str, field: str) -> int:
    if value not in enum:
        raise ValueError(f"Unknown {field} description {value!r} for city {city!r}")
    return enum[value]


def _time_label(timestamp) -> str:
    # string slicing: much faster than pd.to_datetime
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


def _pivot(df: pd.DataFrame, value_col: str, round_ndigits: int | None = None) -> dict:
    """(city, time label, height label) -> value."""
    keys = zip(
        df["city"], df["timestamp"].map(_time_label), df["height"].map(_height_label)
    )
    values = df[value_col]
    if round_ndigits is not None:
        values = values.round(round_ndigits)
    return dict(zip(keys, values))


def _column_group(prefix: str) -> list[str]:
    return [f"{prefix}_{t}_{h}" for t in TIME_GROUPS_ORDER for h in HEIGHT_GROUPS_ORDER]


def _grouped_cells(values: dict, city: str, default) -> list:
    return [
        values.get((city, t, h), default)
        for t in TIME_GROUPS_ORDER
        for h in HEIGHT_GROUPS_ORDER
    ]


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not pd.isna(value)


def get_compass_direction(degrees: float) -> str:
    """Convert a bearing in degrees to an 8-point compass label."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((degrees + 22.5) // 45) % 8
    return directions[idx]


def average_by_height_and_time(lookup: dict, round_ndigits: int = 2) -> dict:
    """
    Groups a (city, time, height)-keyed lookup into (city, time_group,
    height_group), averaging within each group with an arithmetic mean
    that skips missing/non-numeric entries.

    A group with no underlying data for a city has no entry in the returned
    dict, so the rows below fill it with their .get(key, default).
    """
    buckets: dict[tuple, list] = defaultdict(list)
    for (city, t, h), val in sorted(lookup.items()):
        if _is_number(val):
            buckets[(city, _hour_group(t), LEVEL_GROUP_MAP[h])].append(val)
    return {k: round(sum(v) / len(v), round_ndigits) for k, v in buckets.items()}


def average_by_height_and_time_vector(
    speed_lookup: dict, dir_lookup: dict, round_ndigits: int = 2
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
    with each value a dict: {"speed": float, "direction": str}.
    """
    buckets: dict[tuple, list] = defaultdict(list)
    for (city, t, h), speed in sorted(speed_lookup.items()):
        direction = dir_lookup.get((city, t, h))
        if _is_number(speed) and _is_number(direction):
            buckets[(city, _hour_group(t), LEVEL_GROUP_MAP[h])].append(
                (speed, direction)
            )

    result = {}
    for key, vals in buckets.items():
        u = [speed * math.sin(math.radians(d)) for speed, d in vals]
        v = [speed * math.cos(math.radians(d)) for speed, d in vals]
        u_mean = sum(u) / len(u)
        v_mean = sum(v) / len(v)

        resultant_speed = round(math.hypot(u_mean, v_mean), round_ndigits)
        resultant_deg = math.degrees(math.atan2(u_mean, v_mean)) % 360
        resultant_dir = get_compass_direction(resultant_deg)

        result[key] = {"speed": resultant_speed, "direction": resultant_dir}

    return result


def total_cloud_by_time(lookup: dict, round_ndigits: int = 2) -> dict:
    """Column cloud cover (hourly max over the levels) per (city, time group),
    plus the whole "day"."""
    hourly: dict[tuple, list] = defaultdict(list)
    for (city, t, _), val in sorted(lookup.items()):
        if _is_number(val):
            hourly[(city, t)].append(val)
    buckets: dict[tuple, list] = defaultdict(list)
    for (city, t), vals in hourly.items():
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
            if _is_number(v):
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
            "region_temp_fronts",
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
    (("temp_fronts_value",), 0.1),  # K, warm-front contrast of 1-6 K
    (("temperature", "temp_fronts", "lapse"), 1),  # K
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


def _front_columns(prefix: str, groups: list[str]) -> list[str]:
    return [
        *(f"{prefix}_hours_{g}" for g in groups),
        *(f"{prefix}_value_{g}" for g in groups),
        *(f"region_{prefix}_area_{g}" for g in groups),
    ]


def fronts_by_city(
    df: pd.DataFrame, prefix: str, groups: list[str], cities: list[str]
) -> dict:
    """Per city and level group: hours inside a front, mean contrast of the
    front with the background (inside - outside, 0 if in none), and the
    largest regional front area (km2) of an hour and level."""
    value_col = next(c for c in df.columns if c.endswith("_inside"))
    outside_col = value_col.removesuffix("_inside") + "_outside"
    df = df.assign(value=df[value_col] - df[outside_col])
    group = np.array([LEVEL_GROUP_MAP[_height_label(h)] for h in df["height"]])

    out: dict[str, dict] = {city: {} for city in cities}
    for g in groups:
        sub = df[group == g]
        areas = sub.groupby(["timestamp", "height"])["area"].sum()
        region_area = int(areas.max()) if len(areas) else 0
        members = [str(c).split(",") for c in sub["cities"]]
        for city in cities:
            inside = sub[np.array([city in m for m in members], dtype=bool)]
            value = inside["value"].mean()  # NaN when the city is in no front
            out[city][f"{prefix}_hours_{g}"] = int(inside["timestamp"].nunique())
            out[city][f"{prefix}_value_{g}"] = (
                0.0 if pd.isna(value) else round(float(value), 2)
            )
            out[city][f"region_{prefix}_area_{g}"] = region_area
    return out


def threshold_by_city(
    df: pd.DataFrame, prefix: str, groups: list[str], cities: list[str]
) -> dict:
    """Per city and level group: hours the city mean is past the threshold,
    mean value then, and the largest regional area (km2) past it."""
    value_col = df.columns[3]
    group = np.array([LEVEL_GROUP_MAP[_height_label(h)] for h in df["height"]])

    out: dict[str, dict] = {city: {} for city in cities}
    for g in groups:
        sub = df[group == g]
        region_area = int(sub["area"].max()) if len(sub) else 0
        past = sub[sub["past"] == 1]
        for city in cities:
            inside = past[past["city"] == city]
            value = inside[value_col].mean() if len(inside) else 0.0
            out[city][f"{prefix}_hours_{g}"] = int(inside["timestamp"].nunique())
            out[city][f"{prefix}_value_{g}"] = round(float(value), 2)
            out[city][f"region_{prefix}_area_{g}"] = region_area
    return out


# (reader, reasoning files {prefix}.txt, level groups); the dataset columns
# are named after the prefix
FRONT_TABLES = [
    (fronts_by_city, ["humidity_fronts", "temp_fronts"], HEIGHT_GROUPS_ORDER),
    # thresholds only exist at 850/700/500 hPa
    (threshold_by_city, ["front", "warmadv", "tefall", "terise"], ["low", "medium"]),
]

FRONT_COLUMNS = [
    c
    for _, files, groups in FRONT_TABLES
    for p in files
    for c in _front_columns(p, groups)
]


CATEGORICAL_COLUMNS = ["location", *_column_group("wind_direction")]


class FoldRmTranslator(BaseTranslator):
    extension = "csv"

    @staticmethod
    def schema(header: list[str]) -> dict:
        """Splits a dataset header into label, feature (header order),
        categorical and numeric columns."""
        categorical = set(CATEGORICAL_COLUMNS)
        non_features = set(LABEL_COLUMNS) | {DATE_COLUMN}
        features = [c for c in header if c not in non_features]
        return {
            "labels": list(LABEL_COLUMNS),
            "features": features,
            "categorical": [c for c in features if c in categorical],
            "numeric": [c for c in features if c not in categorical],
        }

    @classmethod
    def schema_from_csv(cls, path: Path) -> dict:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        return cls.schema(header)

    def translate_day(self, day_input_folder: Path, target_date: date) -> bytes:
        cities_gt = json.loads((day_input_folder / "gt.json").read_text())

        reasoning_dir = day_input_folder / "reasoning"
        winds_df = pd.read_csv(reasoning_dir / "winds.txt", sep="\t")
        segments_df = pd.read_csv(reasoning_dir / "cloud_segments.txt", sep="\t")
        temp_df = pd.read_csv(reasoning_dir / "temp.txt", sep="\t")
        humidity_df = pd.read_csv(reasoning_dir / "humidity.txt", sep="\t")
        cloud_cover_df = pd.read_csv(reasoning_dir / "cloud.txt", sep="\t")
        featured = set().union(
            *(df["city"] for df in (winds_df, temp_df, humidity_df, cloud_cover_df))
        )
        cities = sorted(set(cities_gt) & featured)
        if not cities:
            raise ValueError(
                f"No city has both ground truth and features: "
                f"{sorted(cities_gt)} vs {sorted(featured)}"
            )

        fronts: dict[str, dict] = {city: {} for city in cities}
        for by_city, files, groups in FRONT_TABLES:
            for prefix in files:
                df = pd.read_csv(reasoning_dir / f"{prefix}.txt", sep="\t")
                for city, values in by_city(df, prefix, groups, cities).items():
                    fronts[city].update(values)

        # several clouds over a city in the same hour add up
        segments_df = (
            segments_df.groupby(["timestamp", "height", "city"]).sum().reset_index()
        )

        wind_dir = _pivot(winds_df, "wind_direction")
        wind_speed = _pivot(winds_df, "wind_speed", round_ndigits=1)

        coverage = _pivot(segments_df, "%covered")
        size = _pivot(segments_df, "tot area")

        temperature = _pivot(temp_df, "temp", round_ndigits=1)
        humidity = _pivot(humidity_df, "humidity", round_ndigits=1)
        cloud_cover = _pivot(cloud_cover_df, "cloud", round_ndigits=1)

        # cloud_segments.txt only has the (city, hour, height) a tobac cloud covers:
        # every other one counts as 0 coverage / size
        all_hours = [f"{h:02d}00" for h in range(24)]
        heights = sorted({h for _, _, h in cloud_cover})
        for key in itertools.product(cities, all_hours, heights):
            coverage.setdefault(key, 0)
            size.setdefault(key, 0)

        wind_grouped = average_by_height_and_time_vector(wind_speed, wind_dir)
        # (column group, (city, time group, height group) -> value, default)
        grouped = [
            (
                "wind_direction",
                {k: v["direction"] for k, v in wind_grouped.items()},
                "",
            ),
            ("wind_speed", {k: v["speed"] for k, v in wind_grouped.items()}, ""),
            ("coverage_clouds", average_by_height_and_time(coverage), 0),
            ("size_cloud", average_by_height_and_time(size), 0),
            ("temperature", average_by_height_and_time(temperature), ""),
            ("humidity", average_by_height_and_time(humidity), ""),
        ]
        cloud_cover_grouped = average_by_height_and_time(cloud_cover)
        cloud_total = total_cloud_by_time(cloud_cover)
        daily, region_daily = daily_predictors(
            humidity, temperature, wind_speed, wind_dir, cloud_cover, cities
        )
        total_groups = [*TIME_GROUPS_ORDER, "day"]

        header = [*LABEL_COLUMNS, "month", "location"]
        for name, _, _ in grouped:
            header += _column_group(name)
        header += FRONT_COLUMNS
        header += _column_group("cloud_cover")
        header += [f"cloud_total_{tg}" for tg in total_groups]
        header += DAILY_COLUMNS + REGION_COLUMNS

        rows = []
        for city in cities:
            gt_entry = cities_gt[city]
            pioggia = _lookup_enum(
                _PIOGGIA_ENUM, gt_entry["PIOGGIA_DESCRIZIONE"], city, "PIOGGIA"
            )
            cloud = _lookup_enum(
                _CLOUD_ENUM, gt_entry["CIELO_DESCRIZIONE"], city, "CIELO"
            )
            row = [f"{pioggia}", f"{cloud}", target_date.month, _slugify_city(city)]
            for _, values, default in grouped:
                row += _grouped_cells(values, city, default)
            row += [fronts[city][c] for c in FRONT_COLUMNS]
            row += _grouped_cells(cloud_cover_grouped, city, "")
            row += [cloud_total.get((city, tg), "") for tg in total_groups]
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

        date_format = "%Y-%m-%d"
        output_filename = (
            f"{min(dates).strftime(date_format)}_"
            f"{max(dates).strftime(date_format)}.{self.extension}"
        )
        output_path = input_folder / output_filename

        dfs = []
        for d in sorted(dates):
            df = pd.read_csv(
                input_folder / f"{d.strftime(date_format)}.{self.extension}"
            )
            date_col = pd.DataFrame({DATE_COLUMN: d.isoformat()}, index=df.index)
            dfs.append(pd.concat([date_col, df], axis=1))

        pd.concat(dfs, ignore_index=True).to_csv(output_path, index=False)
        self.logger.debug(f"Successfully merged {len(dfs)} files into: {output_path}")
        return output_path
