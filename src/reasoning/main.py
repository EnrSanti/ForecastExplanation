import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import xarray as xr
from tqdm import tqdm

from region import Region

from .fronts import detect_phenomenon, detect_phenomenon_fronts
from .segment import detect_clouds, detect_winds
from .utils import get_heights

logger = logging.getLogger("ForecastExplanation")


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
        radius = region.city_radius

        detect_winds(
            feat_ds,
            region.get_cities(),
            day_output_dir / "winds.txt",
            heights,
            radius,
        )

        detect_clouds(
            seg_ds,
            feat_ds,
            region.get_cities(),
            day_output_dir / "cloud.txt",
            heights,
            radius,
        )
        # CERRA cloud cover around each city using raw data
        detect_phenomenon(
            feat_ds,
            region.get_cities(),
            day_output_dir / "cloud_cover.txt",
            heights,
            "cloud",
            radius,
        )

        # Heat
        detect_phenomenon(
            feat_ds,
            region.get_cities(),
            day_output_dir / "heat.txt",
            heights,
            "temp",
            radius,
        )
        detect_phenomenon_fronts(
            seg_ds,
            feat_ds,
            region.get_cities(),
            day_output_dir / "heat_fronts.txt",
            heights,
            "temp",
        )

        # Humidity
        detect_phenomenon(
            feat_ds,
            region.get_cities(),
            day_output_dir / "humidity.txt",
            heights,
            "humidity",
            radius,
        )
        detect_phenomenon_fronts(
            seg_ds,
            feat_ds,
            region.get_cities(),
            day_output_dir / "humidity_fronts.txt",
            heights,
            "humidity",
        )
