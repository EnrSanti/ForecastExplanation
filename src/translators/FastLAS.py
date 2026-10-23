"""FastLAS translator.

translate_day writes one .las file per day with one example per location: its
context is that location's day as qualitative facts (cloud layers, humidity,
wind, fronts, stability, ...) plus the bulletin's (sky, rain) pair.
merge_into_dataset puts a background in front of the days: constant pools,
the region's geography, meteorological concepts derived from the facts (the
vocabulary of the mode bias), and the rules that turn each bulletin into the
example's inclusion and exclusions. The result is a single task:

    FastLAS --opl <first day>_<last day>.las

and the learned rules read like

    forecast(V0,overcast,heavy) :- orographic_lift(V0), deep_moisture(V0), location(V0).
"""

import json
import math
import re
import textwrap
from datetime import date
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

from region import Region

from .fold_rm import (
    FRONT_FILES,
    LEVEL_GROUP_MAP,
    THRESHOLD_FILES,
    TIME_GROUPS_ORDER,
    _city_table,
    _height_label,
    _lookup_enum,
    _pivot,
    _read_tsv_or_empty,
    _slugify_city,
    _theta_e,
    fronts_by_city,
    get_compass_direction,
    threshold_by_city,
    total_cloud_by_time,
)
from .translator import BaseTranslator

_SKY_ATOM = {
    None: "clear",
    False: "clear",
    "sereno": "clear",
    "poco nuvoloso": "few_clouds",
    "variabile": "variable",
    "nuvoloso": "cloudy",
    "coperto": "overcast",
    "sole, nebbia, nubi basse": "fog_low_clouds",
}

_RAIN_ATOM = {
    None: "none",
    False: "none",
    "debole": "light",
    "moderata": "moderate",
    "abbondante": "heavy",
    "intensa": "intense",
    "molto intensa": "very_intense",
}

SKY_VALUES = list(dict.fromkeys(_SKY_ATOM.values()))
RAIN_VALUES = list(dict.fromkeys(_RAIN_ATOM.values()))

# a layer is saturated when its mean RH stays at or above 90 % for 6+ hours,
# dry when its daily mean RH is below 50 %
SATURATED_RH, SATURATED_HOURS, DRY_RH = 90, 6, 50
# daily mean cloud fraction (%) -> amount, by upper bound
CLOUD_AMOUNTS = [
    (15, "none"),
    (50, "scattered"),
    (85, "broken"),
    (math.inf, "overcast"),
]
# mean wind speed (m/s) -> force, by upper bound
WIND_FORCES = [(5, "weak"), (15, "moderate"), (math.inf, "strong")]
# change of column cloud cover (%) across the day that counts as a trend
TREND_CHANGE = 30
SKY_TRENDS = ["steady", "clearing", "thickening", "changeable"]
# stable low levels: 1000 hPa less than 2 K warmer than 925 hPa (about 650 m
# up, so under 3 K/km, well below the moist adiabat; inversions included)
# for 6+ hours. Fog inversions are often too shallow to show between levels
STABLE_LAPSE_K, STABLE_HOURS = 2, 6
# tracked clouds covering half a location for 6+ hours form a cloud system,
# large above 25,000 km2 (about three times the region)
SYSTEM_COVER, SYSTEM_HOURS, LARGE_SYSTEM_KM2 = 50, 6, 25_000
# hours a location must spend inside a front / past a threshold
EVENT_HOURS = 2

# fold_rm column prefix -> predicate, event(Location, Layer)
EVENT_PREDICATES = {
    "front": "front",  # strong thermal gradient
    "warmadv": "warm_advection",
    "terise": "moist_air_inflow",  # sharp rise of theta-e
    "tefall": "cold_air_inflow",  # sharp fall of theta-e
    "temperature_fronts": "warm_air_mass",  # warmer than its surroundings
    "humidity_fronts": "dry_intrusion",  # drier than its surroundings
}

ZONES = {
    "alpi_carniche": "alps",
    "alpi_giulie": "alps",
    "prealpi_carniche": "prealps",
    "prealpi_giulie": "prealps",
    "pordenone": "plain",
    "udine": "plain",
    "gorizia": "plain",
    "lignano": "coast",
    "trieste": "coast",
}
SEASONS = ["winter", "spring", "summer", "autumn"]

# background concepts over a location: the vocabulary of the learned rules
CONCEPTS = [
    "moist_inflow",
    "orographic_lift",
    "deep_moisture",
    "frontal_passage",
    "warm_moist_advection",
    "convective",
    "dry_air_aloft",
    "dry_northerly",
    "fog_risk",
    "thick_clouds",
    "low_clouds_only",
    "high_clouds_only",
    "cloud_band",
]
# the ones the learner may also negate, enough to keep e.g. a sky rule off
# rainy or foggy days. Every negated mode adds a literal to almost every
# example's most specific rule, and FastLAS's run time grows steeply with
# that: these three cost about 5x
NEGATED_CONCEPTS = ["fog_risk", "deep_moisture", "frontal_passage"]

CONCEPT_RULES = """\
% where the air comes from: off the Adriatic (S, SE, SW) or over the Alps (N, NE, NW)
from_sea(L, Y) :- wind_from(L, Y, s).
from_sea(L, Y) :- wind_from(L, Y, se).
from_sea(L, Y) :- wind_from(L, Y, sw).
from_north(L, Y) :- wind_from(L, Y, n).
from_north(L, Y) :- wind_from(L, Y, ne).
from_north(L, Y) :- wind_from(L, Y, nw).
moist(L, Y) :- humidity(L, Y, moist).
moist(L, Y) :- humidity(L, Y, saturated).
cloud_deck(L, Y) :- cloud_cover(L, Y, broken).
cloud_deck(L, Y) :- cloud_cover(L, Y, overcast).

% saturated Adriatic air flowing in at low levels...
moist_inflow(L) :- from_sea(L, low), humidity(L, low, saturated).
% ...and forced up the Prealps and the Alps
orographic_lift(L) :- moist_inflow(L), zone(L, prealps).
orographic_lift(L) :- moist_inflow(L), zone(L, alps).
% saturated from the low to the middle troposphere
deep_moisture(L) :- humidity(L, low, saturated), humidity(L, medium, saturated).
% a front, or colder and drier air replacing the low-level air mass (a fall
% of theta-e aloft alone is more often dry air sinking than a front)
frontal_passage(L) :- front(L, Y).
frontal_passage(L) :- cold_air_inflow(L, low).
% warm, moist air arriving (overrunning ahead of a warm front)
warm_moist_advection(L) :- warm_advection(L, Y), moist(L, medium).
warm_moist_advection(L) :- moist_air_inflow(L, Y).
% a potentially unstable column with moisture to feed showers and storms
convective(L) :- unstable(L), moist(L, low).
% a dry middle troposphere, where clouds struggle to thicken
dry_air_aloft(L) :- humidity(L, medium, dry).
dry_air_aloft(L) :- dry_intrusion(L, medium).
% dry northerly flow (foehn, bora)
dry_northerly(L) :- from_north(L, low), humidity(L, low, dry).
% stagnant saturated air in a stable layer: fog and low stratus
fog_risk(L) :- stable_low_levels(L), humidity(L, low, saturated),
    wind_strength(L, low, weak).
% which cloud layers are there
thick_clouds(L) :- cloud_deck(L, low), cloud_deck(L, medium).
low_clouds_only(L) :- cloud_deck(L, low),
    not cloud_deck(L, medium), not cloud_deck(L, high).
high_clouds_only(L) :- cloud_deck(L, high),
    not cloud_deck(L, low), not cloud_deck(L, medium).
% a large organised cloud system below the cirrus level (a frontal cloud band)
cloud_band(L) :- cloud_system(L, low, large).
cloud_band(L) :- cloud_system(L, medium, large).
"""

# every example's inclusions and exclusions, derived through FastLAS's reserved
# inclusion/1 and exclusion/1 from its bulletin and the outcomes in the data
EXAMPLE_RULES = """\
% the bulletin's (sky, rain) must be forecast, no other outcome of the data may be
inclusion(forecast(L, S, R)) :- bulletin(L, S, R).
exclusion(forecast(L, S, R)) :- location(L), outcome(S, R), not bulletin(L, S, R).
"""


def _atom(predicate: str, *args: str) -> str:
    return f"{predicate}({','.join(args)})" if args else predicate


def _pool(predicate: str, values: list[str]) -> str:
    return " ".join(f"{predicate}({v})." for v in values)


def _background(outcomes: list[tuple[str, str]]) -> str:
    return "\n".join(
        [
            "% ===== Background",
            "% constant pools of the mode bias",
            _pool("sky", SKY_VALUES),
            _pool("rain", RAIN_VALUES),
            _pool("amount", [label for _, label in CLOUD_AMOUNTS]),
            _pool("trend", SKY_TRENDS),
            _pool("zone_type", list(dict.fromkeys(ZONES.values()))),
            _pool("season_name", SEASONS),
            "",
            "% the (sky, rain) outcomes of the bulletins in the data",
            " ".join(f"outcome({sky},{rain})." for sky, rain in outcomes),
            EXAMPLE_RULES,
            "% geography",
            *(f"zone({loc}, {zone})." for loc, zone in ZONES.items()),
            "",
            CONCEPT_RULES,
        ]
    )


MODE_BIAS = "\n".join(
    [
        "% ===== Mode bias",
        "#modeh(forecast(var(location), const(sky), const(rain))).",
        *(f"#modeb({c}(var(location)))." for c in CONCEPTS),
        *(f"#modeb(not {c}(var(location)))." for c in NEGATED_CONCEPTS),
        "#modeb(sky_cover(var(location), const(amount))).",
        "#modeb(sky_trend(var(location), const(trend))).",
        "% each of these adds a literal to every example's most specific rule,",
        "% which costs FastLAS 2-4x the run time:",
        "% #modeb(zone(var(location), const(zone_type))).",
        "% #modeb(season(const(season_name))).",
        "#maxv(1).",
    ]
)

# shortest hypothesis: one point per literal
SCORING = "\n".join(
    [
        "% ===== Scoring",
        '#bias("penalty(1, head) :- in_head(X).").',
        '#bias("penalty(1, body(X)) :- in_body(X).").',
    ]
)


def _bin(value: float, bins: list[tuple[float, str]]) -> str:
    return next(label for upper, label in bins if value < upper)


def _season(month: int) -> str:
    return SEASONS[month % 12 // 3]


def _with_layer(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(layer=[LEVEL_GROUP_MAP[_height_label(h)] for h in df["height"]])


def _by_level(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """(city, timestamp) x pressure level ("0850", ...) table of a column."""
    levels = [_height_label(h) for h in df["height"]]
    return df.assign(level=levels).pivot_table(
        index=["city", "timestamp"], columns="level", values=column
    )


def _sky_trend(parts: list[float]) -> str:
    """Trend of the column cloud cover over the parts of the day, in order."""
    if max(parts) - min(parts) < TREND_CHANGE:
        return "steady"
    if parts[-1] - parts[0] >= TREND_CHANGE:
        return "thickening"
    if parts[0] - parts[-1] >= TREND_CHANGE:
        return "clearing"
    return "changeable"


def cloud_facts(cloud_cover_df: pd.DataFrame, cities: list[str]):
    """sky_cover(L, amount) and sky_trend(L, trend) from the column cover
    (fold_rm's cloud_total), cloud_cover(L, layer, amount) from the daily mean
    of the hourly layer cover (the max over the layer's levels)."""
    lookup, times, heights = _pivot(cloud_cover_df, "cloud")
    totals = total_cloud_by_time(lookup, cities, times, heights)
    for city in cities:
        if (city, "day") in totals:
            yield city, ("sky_cover", _bin(totals[(city, "day")], CLOUD_AMOUNTS))
            parts = [
                totals[city, tg] for tg in TIME_GROUPS_ORDER if (city, tg) in totals
            ]
            yield city, ("sky_trend", _sky_trend(parts))

    hourly = (
        _with_layer(cloud_cover_df)
        .groupby(["city", "layer", "timestamp"])["cloud"]
        .max()
    )
    for (city, layer), cover in hourly.groupby(level=["city", "layer"]).mean().items():
        yield city, ("cloud_cover", layer, _bin(cover, CLOUD_AMOUNTS))


def humidity_facts(humidity_df: pd.DataFrame):
    """humidity(L, layer, dry|moist|saturated) from the hourly layer mean RH."""
    hourly = (
        _with_layer(humidity_df)
        .groupby(["city", "layer", "timestamp"])["humidity"]
        .mean()
    )
    for (city, layer), rh in hourly.groupby(level=["city", "layer"]):
        if (rh >= SATURATED_RH).sum() >= SATURATED_HOURS:
            level = "saturated"
        elif rh.mean() < DRY_RH:
            level = "dry"
        else:
            level = "moist"
        yield city, ("humidity", layer, level)


def wind_facts(winds_df: pd.DataFrame):
    """wind_from(L, layer, compass) from the day's resultant wind vector,
    wind_strength(L, layer, force) from the mean speed."""
    rad = np.radians(winds_df["wind_direction"])
    winds = _with_layer(winds_df).assign(
        u=winds_df["wind_speed"] * np.sin(rad),
        v=winds_df["wind_speed"] * np.cos(rad),
    )
    for (city, layer), g in winds.groupby(["city", "layer"]):
        bearing = math.degrees(math.atan2(g["u"].mean(), g["v"].mean())) % 360
        yield city, ("wind_from", layer, get_compass_direction(bearing).lower())
        yield city, ("wind_strength", layer, _bin(g["wind_speed"].mean(), WIND_FORCES))


def event_facts(reasoning_dir: Path, cities: list[str]):
    """front(L, layer), warm_advection(L, layer), ...: the location is inside
    the phenomenon (fold_rm's *_hours_* columns) at that layer."""
    for files, by_city, columns in (
        (FRONT_FILES, fronts_by_city, ["timestamp", "height", "area", "cities"]),
        (THRESHOLD_FILES, threshold_by_city, ["timestamp", "height", "city"]),
    ):
        for file_name, (prefix, groups) in files.items():
            df = _read_tsv_or_empty(reasoning_dir / file_name, columns)
            for city, values in by_city(df, prefix, groups, cities).items():
                for g in groups:
                    if values[f"{prefix}_hours_{g}"] >= EVENT_HOURS:
                        yield city, (EVENT_PREDICATES[prefix], g)


def stability_facts(heat_df: pd.DataFrame, humidity_df: pd.DataFrame):
    """stable_low_levels(L): weak or inverted lapse rate from 1000 to 925 hPa.
    unstable(L): theta-e falls from 850 to 500 hPa at some hour."""
    t = _by_level(heat_df, "temperature")
    rh = _by_level(humidity_df, "humidity").reindex(t.index)
    if {"0925", "1000"} <= set(t.columns):
        stable = t["1000"] - t["0925"] < STABLE_LAPSE_K
        hours = stable.groupby(level="city").sum()
        for city in hours[hours >= STABLE_HOURS].index:
            yield city, ("stable_low_levels",)
    if {"0850", "0500"} <= set(t.columns) & set(rh.columns):
        theta_e = {
            lvl: pd.Series(
                [_theta_e(a, b, float(lvl)) for a, b in zip(t[lvl], rh[lvl])],
                index=t.index,
            )
            for lvl in ("0850", "0500")
        }
        drop = (theta_e["0850"] - theta_e["0500"]).groupby(level="city").max()
        for city in drop[drop > 0].index:
            yield city, ("unstable",)


def cloud_system_facts(cloud_df: pd.DataFrame):
    """cloud_system(L, layer, small|large) from the tracked clouds."""
    # several clouds over a city in the same hour add up
    cloud = (
        cloud_df.groupby(["timestamp", "height", "city"])[["tot area", "%covered"]]
        .sum()
        .reset_index()
    )
    cloud = _with_layer(cloud[cloud["%covered"] >= SYSTEM_COVER])
    for (city, layer), g in cloud.groupby(["city", "layer"]):
        if g["timestamp"].nunique() >= SYSTEM_HOURS:
            size = "large" if g["tot area"].mean() >= LARGE_SYSTEM_KM2 else "small"
            yield city, ("cloud_system", layer, size)


def _example(example_id: str, context: list[str], penalty: int) -> str:
    """One location-day. Its inclusion and exclusions come from the bulletin/3
    fact in the context (see EXAMPLE_RULES)."""
    facts = textwrap.fill(
        " ".join(f"{c}." for c in context),
        width=100,
        initial_indent="    ",
        subsequent_indent="    ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    return f"#pos({example_id}@{penalty}, {{}}, {{}}, {{\n{facts}\n}})."


class FastLasTranslator(BaseTranslator):
    extension = "LAS"
    # cost of leaving one location-day unexplained, against one point per
    # rule literal: raise it for more (and more specific) rules
    example_penalty = 1

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
        cloud_df = _read_tsv_or_empty(
            reasoning_dir / "cloud.txt",
            ["timestamp", "height", "cloud_id", "tot area", "city", "%covered"],
        )
        heat_df = _city_table(reasoning_dir / "heat.txt", "temperature", city_coords)
        humidity_df = _city_table(
            reasoning_dir / "humidity.txt", "humidity", city_coords
        )
        cloud_cover_df = _city_table(
            reasoning_dir / "cloud_cover.txt", "cloud", city_coords
        )

        facts: dict[str, list[tuple]] = {city: [] for city in cities}
        for city, fact in chain(
            cloud_facts(cloud_cover_df, cities),
            humidity_facts(humidity_df),
            wind_facts(winds_df),
            event_facts(reasoning_dir, cities),
            stability_facts(heat_df, humidity_df),
            cloud_system_facts(cloud_df),
        ):
            if city in facts:
                facts[city].append(fact)

        blocks = [f"% ===== {target_date.isoformat()}"]
        for city in cities:
            loc = _slugify_city(city)
            if loc not in ZONES:
                self.logger.warning(
                    f"No zone for {city!r}: no zone/2 fact in the background"
                )
            gt_entry = cities_gt[city]
            sky = _lookup_enum(
                _SKY_ATOM, gt_entry.get("CIELO_DESCRIZIONE"), city, "CIELO"
            )
            rain = _lookup_enum(
                _RAIN_ATOM, gt_entry.get("PIOGGIA_DESCRIZIONE"), city, "PIOGGIA"
            )
            context = [
                _atom("bulletin", loc, sky, rain),
                _atom("location", loc),
                _atom("season", _season(target_date.month)),
                *(_atom(name, loc, *args) for name, *args in facts[city]),
            ]
            blocks.append(f"% {city}: sky {sky}, rain {rain}")
            blocks.append(
                _example(f"d{target_date:%Y%m%d}_{loc}", context, self.example_penalty)
            )
        return ("\n".join(blocks) + "\n").encode("utf-8")

    def merge_into_dataset(self, dates: list[date], input_folder: Path) -> Path | None:
        if not dates:
            self.logger.error("No dates provided.")
            return None

        input_path = Path(input_folder)

        # Determine the smallest and biggest dates for the output filename
        min_date = min(dates)
        max_date = max(dates)

        date_format = "%Y-%m-%d"
        output_filename = (
            f"{min_date.strftime(date_format)}_{max_date.strftime(date_format)}"
            f".{self.extension}"
        )
        output_path = input_path / output_filename

        # duplicated dates would duplicate example ids, which FastLAS rejects
        days = []
        for d in sorted(set(dates)):
            file_path = input_path / f"{d.strftime(date_format)}.{self.extension}"
            if file_path.exists():
                days.append(file_path.read_text())
            else:
                self.logger.warning(f"File not found for date {file_path.name}")

        if not days:
            self.logger.error("No matching LAS files found for the given dates.")
            return None

        # outcomes no bulletin uses cannot be learned, and excluding them
        # would only slow FastLAS down
        seen = set(re.findall(r"bulletin\(\w+,(\w+),(\w+)\)", "".join(days)))
        outcomes = [(s, r) for s in SKY_VALUES for r in RAIN_VALUES if (s, r) in seen]

        header = (
            "% FastLAS task: forecast(Location, Sky, Rain) for every location\n"
            f"% and day, {min_date.isoformat()} to {max_date.isoformat()}.\n"
            f"% Run with: FastLAS --opl {output_filename}"
        )
        output_path.write_text(
            "\n\n".join([header, _background(outcomes), MODE_BIAS, SCORING, *days])
        )
        self.logger.debug(f"Successfully merged {len(days)} files into: {output_path}")
        return output_path
