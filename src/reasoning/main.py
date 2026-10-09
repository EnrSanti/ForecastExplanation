import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import xarray as xr
from tqdm import tqdm

from region import Region

from .fronts import detect_phenomenon, detect_phenomenon_fronts, detect_threshold
from .segment import detect_clouds, detect_winds
from .utils import get_heights

logger = logging.getLogger("ForecastExplanation")

# (output file, raw field)
RAW_TABLES = [("cloud_cover", "cloud"), ("heat", "temp"), ("humidity", "humidity")]

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

    ok = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _reason_single_day,
                target_date,
                input_dir,
                output_dir,
                region,
                force=force,
            ): target_date
            for target_date in dates
        }

        for future in tqdm(as_completed(futures), total=len(dates), desc="Reasoning"):
            target_date = futures[future]
            try:
                future.result()
                ok.append(target_date)
            except Exception:
                logger.exception(f"Reasoning failed for {target_date}")

    logger.info("Reasoning completed.")
    return sorted(ok)


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
        heights = get_heights(feat_ds)
        cities = region.get_cities()
        radius = region.city_radius

        detect_winds(feat_ds, cities, day_output_dir / "winds.txt", heights, radius)
        detect_clouds(
            seg_ds, feat_ds, cities, day_output_dir / "cloud.txt", heights, radius
        )

        # mean raw values around each city
        for name, phenomenon in RAW_TABLES:
            detect_phenomenon(
                feat_ds,
                cities,
                day_output_dir / f"{name}.txt",
                heights,
                phenomenon,
                radius,
            )

        # tobac segments of the warmest / driest areas
        for name, phenomenon in (("heat", "temp"), ("humidity", "humidity")):
            detect_phenomenon_fronts(
                seg_ds,
                feat_ds,
                cities,
                day_output_dir / f"{name}_fronts.txt",
                heights,
                phenomenon,
            )

        for phenomenon, field, sign, thresholds in FRONT_PHENOMENA:
            detect_threshold(
                feat_ds,
                cities,
                day_output_dir / f"{phenomenon}_fronts.txt",
                field,
                thresholds,
                sign,
                radius,
            )
