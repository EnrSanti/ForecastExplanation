import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List

import xarray as xr
from tqdm import tqdm

from . import Region, RAW_DATA_DIR, CUT_DATA_DIR, DISCRETE_DATA_DIR, CLUSTERED_DATA_DIR
from .extract_features_nc import (
    create_one_time_images,
    build_feature_dataarrays,
)
from .get_raw_data import extract_nc
from .image_proc import cluster_xarray

logger = logging.getLogger(__name__)


def find_starting_step(date: datetime) -> int:
    clustered_dir = os.path.join(CLUSTERED_DATA_DIR, date.strftime("%Y-%m-%d"))
    raw_data_dir = os.path.join(RAW_DATA_DIR, date.strftime("%Y-%m-%d"))
    cut_data_dir = os.path.join(CUT_DATA_DIR, date.strftime("%Y-%m-%d"))
    discrete_data_dir = os.path.join(DISCRETE_DATA_DIR, date.strftime("%Y-%m-%d"))

    if os.path.exists(clustered_dir) and len(os.listdir(clustered_dir)) > 0:
        return 4  # Clustering done
    elif os.path.exists(discrete_data_dir) and len(os.listdir(discrete_data_dir)) > 0:
        return 3  # Feature maps saved
    elif os.path.exists(cut_data_dir) and len(os.listdir(cut_data_dir)) > 0:
        return 2  # GRIB cut
    elif os.path.exists(raw_data_dir) and len(os.listdir(raw_data_dir)) > 0:
        return 1  # GRIB downloaded
    else:
        return 0  # No data


def extract_day_worker(
    date,
    region,
    clean_level: int = 0,
    clustering: bool = True,
    force_redo: int = 0,
    stopping_step: int = 4,
):
    logger.debug(f"Extracting data for {date.strftime('%Y-%m-%d')}")
    clustered_dir = os.path.join(CLUSTERED_DATA_DIR, date.strftime("%Y-%m-%d"))
    raw_data_dir = os.path.join(RAW_DATA_DIR, date.strftime("%Y-%m-%d"))
    cut_data_dir = os.path.join(CUT_DATA_DIR, date.strftime("%Y-%m-%d"))
    discrete_data_dir = os.path.join(DISCRETE_DATA_DIR, date.strftime("%Y-%m-%d"))
    features_nc_path = os.path.join(discrete_data_dir, "features.nc")

    starting_step = find_starting_step(date)
    nc_file = ""

    # todo better stepping
    if starting_step in [0, 1, 2] or force_redo >= 2:
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(cut_data_dir, exist_ok=True)
        # this skips 0 1 automatically if already done
        nc_file = extract_nc(date, region, raw_data_dir, cut_data_dir, force_redo)
        starting_step = 2

    if (starting_step == 2 or force_redo >= 2) and stopping_step >= 3:
        os.makedirs(discrete_data_dir, exist_ok=True)
        feature_data = build_feature_dataarrays(nc_file)
        feature_data.to_netcdf(features_nc_path)
        starting_step = 3

    if ((starting_step == 3 or force_redo >= 1) and clustering) and stopping_step >= 4:
        os.makedirs(clustered_dir, exist_ok=True)
        with xr.open_dataset(features_nc_path) as features_ds:
            feature_data = {str(name): da for name, da in features_ds.data_vars.items()}
        clustered_data = cluster_xarray(feature_data)
        clustered_data.to_netcdf(os.path.join(clustered_dir, "features.nc"))

    if starting_step == 4:
        logger.info(
            f"Feature maps already exist for {date.strftime('%Y-%m-%d')}, skipping."
        )

    if clean_level >= 1:
        shutil.rmtree(raw_data_dir, ignore_errors=True)
    if clean_level >= 2:
        shutil.rmtree(cut_data_dir, ignore_errors=True)
    if clean_level >= 3:
        shutil.rmtree(discrete_data_dir, ignore_errors=True)
        shutil.rmtree(clustered_dir, ignore_errors=True)


def extract_day(
    dates: List[datetime],
    region: Region,
    clean_level: int = 0,
    clustering: bool = True,
    force_redo: int = 0,
    stopping_step: int = 4,
) -> None:
    logger.info("Starting data extraction...")

    worker = extract_day_worker

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(
                worker, date, region, clean_level, clustering, force_redo, stopping_step
            ): date
            for date in dates
        }

        for future in tqdm(
            as_completed(futures), total=len(dates), desc="Data Extraction"
        ):
            date = futures[future]
            try:
                future.result()
            except Exception:
                logger.error(f"Extract failed for {date}", exc_info=True)

    logger.info("Data extraction completed.")


def extract(
    dates: List[datetime],
    region: Region,
    clean_level: int = 0,
    clustering: bool = True,
    force_redo: int = 0,
    just_cut: bool = False,
    create_images: bool = False,  # todo handle this
) -> None:
    os.makedirs(CLUSTERED_DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    os.makedirs(CUT_DATA_DIR, exist_ok=True)
    os.makedirs(DISCRETE_DATA_DIR, exist_ok=True)

    if create_images:
        create_one_time_images(region, DISCRETE_DATA_DIR)
    stopping_step = 4
    if just_cut:
        stopping_step = 1

    extract_day(
        dates,
        region,
        clean_level,
        clustering,
        force_redo,
        stopping_step=stopping_step,
    )
