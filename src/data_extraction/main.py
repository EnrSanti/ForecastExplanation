import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from . import (
    CLUSTERED_DATA_DIR,
    CUT_DATA_DIR,
    DISCRETE_DATA_DIR,
    RAW_DATA_DIR,
    Region,
)
from .clustering import cluster_xarray
from .extract_features_nc import (
    build_feature_dataarrays,
    create_one_time_images,
)
from .get_raw_data import cut_month, cut_path, to_compressed_netcdf

logger = logging.getLogger(__name__)


def find_starting_step(
    clustered_dir: Path, cut_data_dir: Path, discrete_data_dir: Path
) -> int:
    """Find the starting step for the data extraction process."""
    if clustered_dir.exists() and any(clustered_dir.iterdir()):
        return 4  # Clustering done
    if discrete_data_dir.exists() and any(discrete_data_dir.iterdir()):
        return 3  # Feature maps saved
    if cut_data_dir.exists() and any(cut_data_dir.iterdir()):
        return 2  # GRIB cut

    return 0  # No data


def extract_day_worker(
    target_date: date,
    region: Region,
    base_path: Path,
    clustering: bool = True,
    force: bool = False,
    create_images: bool = False,
) -> None:
    logger.debug(f"Extracting data for {target_date.strftime('%Y-%m-%d')}")
    clustered_dir = base_path / CLUSTERED_DATA_DIR / target_date.strftime("%Y-%m-%d")
    cut_data_dir = base_path / CUT_DATA_DIR / target_date.strftime("%Y-%m-%d")
    discrete_data_dir = base_path / DISCRETE_DATA_DIR / target_date.strftime("%Y-%m-%d")
    features_nc_path = discrete_data_dir / "features.nc"
    if force:
        shutil.rmtree(discrete_data_dir, ignore_errors=True)
        shutil.rmtree(clustered_dir, ignore_errors=True)

    starting_step = find_starting_step(clustered_dir, cut_data_dir, discrete_data_dir)

    if starting_step in [0, 2]:
        nc_file = cut_path(target_date, region, base_path)
        if not nc_file.exists():
            raise FileNotFoundError(f"Missing cut file: {nc_file}")
        discrete_data_dir.mkdir(parents=True, exist_ok=True)
        feature_data = build_feature_dataarrays(nc_file)
        to_compressed_netcdf(feature_data, features_nc_path)
        if create_images:
            save_tobac_input_images(feature_data, discrete_data_dir)
        starting_step = 3

    if starting_step == 3 and clustering:
        clustered_dir.mkdir(parents=True, exist_ok=True)
        with xr.open_dataset(features_nc_path, engine="h5netcdf") as features_ds:
            feature_data = {str(name): da for name, da in features_ds.data_vars.items()}
        clustered_data = cluster_xarray(feature_data)
        to_compressed_netcdf(clustered_data, clustered_dir / "features.nc")

    if starting_step == 4:
        logger.debug(
            f"Feature maps already exist for {target_date.strftime('%Y-%m-%d')}, skipping."
        )


def clean_artifacts(dates: list[date], output_path: Path, targets: list[str]) -> None:
    if "grib" in targets:
        logger.debug(
            f"Deleting GRIBs from the shared '{RAW_DATA_DIR}', other runs will re-download them."
        )

    for target_date in dates:
        day = target_date.strftime("%Y-%m-%d")
        dirs = []
        if "grib" in targets:
            dirs.append(Path(RAW_DATA_DIR) / day)
        if "cut" in targets:
            dirs.append(output_path / CUT_DATA_DIR / day)
        if "extracted" in targets:
            dirs.append(output_path / DISCRETE_DATA_DIR / day)
            dirs.append(output_path / CLUSTERED_DATA_DIR / day)
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)

    logger.info(f"Cleaned {', '.join(targets)}")


def save_tobac_input_images(feature_data: xr.Dataset, output_dir: Path) -> None:
    """
    One grayscale PNG per (variable, time):
        output_dir/<variable>/<variable>_<YYYYMMDD_HHMM>.png
    scaled over the whole day of the variable, so a flat frame stays dark
    instead of being stretched to [0, 255].
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for var_name, da in feature_data.data_vars.items():
        var_dir = output_dir / str(var_name)
        var_dir.mkdir(parents=True, exist_ok=True)

        vmin = float(da.min())
        vmax = float(da.max())
        vrange = vmax - vmin if vmax > vmin else 1.0

        for t in range(da.sizes["time"]):
            frame = da.isel(time=t)
            ts = pd.to_datetime(frame["time"].values)
            norm = ((frame.values - vmin) / vrange * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(var_dir / f"{var_name}_{ts:%Y%m%d_%H%M}.png"), norm)

    logger.info(f"Saved TOBAC input images to '{output_dir}'.")


def extract_day(
    dates: list[date],
    region: Region,
    base_path: Path,
    clustering: bool = True,
    force: bool = False,
    create_images: bool = False,
    workers: int = 12,
) -> list[date]:
    logger.info("Starting data extraction...")

    ok = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                extract_day_worker,
                target_date,
                region,
                base_path,
                clustering,
                force,
                create_images,
            ): target_date
            for target_date in dates
        }

        for future in tqdm(
            as_completed(futures), total=len(dates), desc="Data Extraction"
        ):
            target_date = futures[future]
            try:
                future.result()
                ok.append(target_date)
            except Exception:
                logger.exception(f"Extract failed for {target_date}")

    logger.info("Data extraction completed.")
    return sorted(ok)


def cerra_download(
    dates: list[date],
    region: Region,
    output_path: Path,
    force_redo: bool,
    delete_grib: bool,
) -> list[date]:
    """Downloads and cuts the CERRA data for the specified dates and region, saving the results to the output path."""
    months = [
        list(month_dates)
        for _, month_dates in groupby(sorted(dates), key=lambda d: (d.year, d.month))
    ]
    cut_ok = []
    for month_dates in tqdm(months, desc="Download+cut (monthly)"):
        try:
            cut_ok += cut_month(
                month_dates,
                region,
                Path(RAW_DATA_DIR),
                output_path,
                force_redo,
                delete_grib,
            )
        except Exception:
            logger.exception(f"Download/cut failed for {month_dates[0]:%Y-%m}")
    return sorted(cut_ok)


def extract(
    dates: list[date],
    region: Region,
    output_path: Path,
    clustering: bool = True,
    force_cut: bool = False,
    force_extract: bool = False,
    just_cut: bool = False,
    create_images: bool = False,
    workers: int = 12,
    delete_grib: bool = False,
) -> list[date]:
    output_path.mkdir(parents=True, exist_ok=True)
    if create_images and not just_cut:
        (output_path / "legends").mkdir(parents=True, exist_ok=True)
        create_one_time_images(region, output_path / "legends")

    cut_ok = cerra_download(dates, region, output_path, force_cut, delete_grib)

    if just_cut:
        return cut_ok

    return extract_day(
        cut_ok,
        region,
        output_path,
        clustering,
        force_extract,
        create_images=create_images,
        workers=workers,
    )
