import csv
import io
import json
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


class FoldRmTranslator(BaseTranslator):
    extension = "csv"

    def translate_day(self, day_input_folder: Path) -> bytes:
        gt_data = json.loads((day_input_folder / "gt.json").read_text())
        cities_gt = next(iter(gt_data.values()))

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
        temperature, temp_t, temp_h = _pivot(heat_df, "temperature", round_ndigits=1)
        humidity, humidity_t, humidity_h = _pivot(
            humidity_df, "humidity", round_ndigits=1
        )

        header = ["prev_pioggia", "prev_cloud"]
        header += _column_group("wind_direction", wind_dir_t, wind_dir_h)
        header += _column_group("wind_speed", wind_speed_t, wind_speed_h)
        header += _column_group("%coverage_clouds", cloud_t, cloud_h)
        header += _column_group("size_cloud", cloud_t, cloud_h)
        header += _column_group("temperature", temp_t, temp_h)
        header += _column_group("humidity", humidity_t, humidity_h)

        rows = []
        for city in sorted(cities_gt):
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
                wind_dir.get((city, t, h), "") for t in wind_dir_t for h in wind_dir_h
            ]
            row += [
                wind_speed.get((city, t, h), "")
                for t in wind_speed_t
                for h in wind_speed_h
            ]
            row += [coverage.get((city, t, h), 0) for t in cloud_t for h in cloud_h]
            row += [size.get((city, t, h), 0) for t in cloud_t for h in cloud_h]
            row += [temperature.get((city, t, h), "") for t in temp_t for h in temp_h]
            row += [
                humidity.get((city, t, h), "") for t in humidity_t for h in humidity_h
            ]

            rows.append(row)

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(header)
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")
