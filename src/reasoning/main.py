import logging
from datetime import date
from pathlib import Path

import xarray as xr

from region import Region
from utils import haversine, run_days

from .fronts import detect_phenomenon, detect_phenomenon_fronts, detect_threshold
from .segment import detect_clouds, detect_winds
from .utils import get_heights

logger = logging.getLogger("ForecastExplanation")

# raw fields averaged around each city, written to {field}.txt
RAW_FIELDS = ["cloud", "temp", "humidity"]
# segmented fields of the warmest / driest areas, written to {field}_fronts.txt
FRONT_FIELDS = ["temp", "humidity"]

# (name, raw field, sign, threshold per height): beyond = sign * value > threshold
FRONT_PHENOMENA = [
    ("front", "front", 1, {"0850m": 7.0, "0700m": 5.0, "0500m": 3.0}),
    ("warmadv", "tadv", 1, {"0850m": 0.3, "0700m": 0.3, "0500m": 0.3}),
    ("tefall", "te_change", -1, {"0850m": 1.95, "0700m": 1.65, "0500m": 1.5}),
    ("terise", "te_change", 1, {"0850m": 1.95, "0700m": 1.65, "0500m": 1.5}),
]


def reason(
    dates: list[date],
    input_dir: Path,
    output_dir: Path,
    region: Region,
    force: bool = False,
    workers: int = 12,
) -> list[date]:
    """
    Perform reasoning on the input data (nc format) and save the results to the output path (text format).
    Converts raw data to reasoning data, ready to be converted to ASP formats.

    Args:
        dates (list): List of dates for which reasoning is to be performed.
        input_dir (Path): Path to the input directory containing spatial data.
        output_dir (Path): Path to the directory where processed output will be written.
        region (Region): The specific geographic region to be used.
        force (bool, optional): If True, forces the processing of all dates. Defaults to False.
        workers (int, optional): Number of parallel worker processes. Defaults to 12.

    Returns:
        list[date]: The days processed without errors.
    """
    logger.info("Starting reasoning")
    ok = run_days(
        _reason_single_day,
        dates,
        input_dir,
        output_dir,
        region,
        force=force,
        workers=workers,
        desc="Reasoning",
    )
    logger.info("Reasoning completed.")
    return ok


def _check_variables(
    seg_ds: xr.Dataset, feat_ds: xr.Dataset, heights: list[str]
) -> tuple[list[str], list[tuple]]:
    """Warns once about the variables missing from the day's data. Returns the
    heights where every per-height variable is present, and FRONT_PHENOMENA
    restricted to the levels whose raw field is present."""
    seg_fields = ["cloud", *FRONT_FIELDS]
    feat_fields = ["wind", "wind_direction", *(f"raw_{f}" for f in RAW_FIELDS)]
    missing = []
    available = []
    for h in heights:
        needed = [(seg_ds, f"{f}_at_{h}") for f in seg_fields]
        needed += [(feat_ds, f"{f}_at_{h}") for f in feat_fields]
        absent = [v for ds, v in needed if v not in ds]
        missing += absent
        if not absent:
            available.append(h)

    phenomena = []
    for name, field, sign, thresholds in FRONT_PHENOMENA:
        present = {}
        for h, threshold in thresholds.items():
            if f"raw_{field}_at_{h}" in feat_ds:
                present[h] = threshold
            else:
                missing.append(f"raw_{field}_at_{h}")
        phenomena.append((name, field, sign, present))

    if missing:
        logger.warning(
            f"Missing variable(s), skipping their heights: {sorted(set(missing))}"
        )
    return available, phenomena


def _reason_single_day(
    target_date: date,
    input_dir: Path,
    output_dir: Path,
    region: Region,
    force: bool = False,
) -> None:
    day_input_dir = input_dir / target_date.strftime("%Y-%m-%d")
    day_output_dir = output_dir / target_date.strftime("%Y-%m-%d") / "reasoning"

    if not force and day_output_dir.exists() and any(day_output_dir.iterdir()):
        logger.debug(
            f"Reasoning already exists for {target_date.strftime('%Y-%m-%d')}. Skipping."
        )
        return

    day_output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Processing reasoning for {target_date.strftime('%Y-%m-%d')}")

    with (
        xr.open_dataset(day_input_dir / "segmentation.nc", engine="h5netcdf") as seg_ds,
        xr.open_dataset(day_input_dir / "features.nc", engine="h5netcdf") as feat_ds,
    ):
        heights, phenomena = _check_variables(seg_ds, feat_ds, get_heights(feat_ds))
        cities = region.get_cities()
        lats, lons = feat_ds.latitude.values, feat_ds.longitude.values
        # grid points within the radius of each city, for the cities that have any
        city_masks = {
            name: mask
            for name, lat, lon in cities
            if (mask := haversine(lat, lon, lats, lons) <= region.city_radius).any()
        }
        if not city_masks:
            raise ValueError(
                f"No city of {region} has a grid point within {region.city_radius} km"
            )

        detect_winds(feat_ds, city_masks, day_output_dir / "winds.txt", heights)
        detect_clouds(
            seg_ds, feat_ds, city_masks, day_output_dir / "cloud_segments.txt", heights
        )

        # mean raw values around each city
        for phenomenon in RAW_FIELDS:
            detect_phenomenon(
                feat_ds,
                city_masks,
                day_output_dir / f"{phenomenon}.txt",
                heights,
                phenomenon,
            )

        for phenomenon in FRONT_FIELDS:
            detect_phenomenon_fronts(
                seg_ds,
                feat_ds,
                cities,
                day_output_dir / f"{phenomenon}_fronts.txt",
                heights,
                phenomenon,
            )

        for phenomenon, field, sign, thresholds in phenomena:
            detect_threshold(
                feat_ds,
                city_masks,
                day_output_dir / f"{phenomenon}.txt",
                field,
                thresholds,
                sign,
            )
