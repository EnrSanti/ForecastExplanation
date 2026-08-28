import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List

from tqdm import tqdm

from . import Region, RAW_DATA_DIR, CUT_DATA_DIR, DISCRETE_DATA_DIR, CLUSTERED_DATA_DIR
from .extract_features_nc import (
    create_one_time_images,
    save_feature_maps,
    render_feature_maps,
    save_wind_vectors,
)
from .get_raw_data import extract_nc, extract_nc_in_memory
from .image_proc import cluster

logger = logging.getLogger(__name__)


def find_starting_step(date: datetime, region: Region) -> int:
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
    date, region, clean_level: int = 0, clustering: bool = True, force_redo: int = 0
):
    logger.debug(f"Extracting data for {date.strftime('%Y-%m-%d')}")
    clustered_dir = os.path.join(CLUSTERED_DATA_DIR, date.strftime("%Y-%m-%d"))
    raw_data_dir = os.path.join(RAW_DATA_DIR, date.strftime("%Y-%m-%d"))
    cut_data_dir = os.path.join(CUT_DATA_DIR, date.strftime("%Y-%m-%d"))
    discrete_data_dir = os.path.join(DISCRETE_DATA_DIR, date.strftime("%Y-%m-%d"))

    starting_step = find_starting_step(date, region)
    nc_file = ""

    if starting_step in [0, 1, 2] or force_redo >= 2:
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(cut_data_dir, exist_ok=True)
        nc_file = extract_nc(
            date, region, raw_data_dir, cut_data_dir, force_redo
        )  # this skips 0 1 automatically if already done
        starting_step = 2

    if starting_step == 2 or force_redo >= 2:
        os.makedirs(discrete_data_dir, exist_ok=True)
        save_feature_maps(nc_file, region, discrete_data_dir)
        starting_step = 3

    if (starting_step == 3 or force_redo >= 1) and clustering:
        os.makedirs(clustered_dir, exist_ok=True)
        cluster(
            output_dir=clustered_dir,
            label_dir=DISCRETE_DATA_DIR,
            input_dir=discrete_data_dir,
        )

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


def extract_day_worker_in_memory(
    date: datetime,
    region: Region,
    clean_level: int = 0,
    clustering: bool = True,
    force_redo: int = 0,
) -> None:
    logger.debug(f"Extracting data for {date.strftime('%Y-%m-%d')}")
    """In-memory pipeline: GRIB -> xr.Dataset -> Dict[images] -> cluster -> disk output."""
    clustered_dir = os.path.join(CLUSTERED_DATA_DIR, date.strftime("%Y-%m-%d"))
    raw_data_dir = os.path.join(RAW_DATA_DIR, date.strftime("%Y-%m-%d"))

    os.makedirs(clustered_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)

    ds = extract_nc_in_memory(date, region, raw_data_dir)
    images = render_feature_maps(ds, region)
    save_wind_vectors(ds, region, clustered_dir)
    ds.close()
    if clustering:
        cluster(
            output_dir=clustered_dir, label_dir=DISCRETE_DATA_DIR, images_dict=images
        )

    if clean_level >= 1:
        shutil.rmtree(raw_data_dir, ignore_errors=True)


def extract_day(
    dates: List[datetime],
    region: Region,
    clean_level: int = 0,
    in_memory: bool = False,
    clustering: bool = True,
    force_redo: int = 0,
) -> None:
    logger.info("Starting data extraction...")

    worker = extract_day_worker_in_memory if in_memory else extract_day_worker

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(
                worker, date, region, clean_level, clustering, force_redo
            ): date
            for date in dates
        }

        for future in tqdm(
            as_completed(futures), total=len(dates), desc="Data Extraction"
        ):
            date = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Extract failed for {date}", exc_info=True)

    logger.info("Data extraction completed.")


def extract(
    dates: List[datetime],
    region: Region,
    clean_level: int = 0,
    in_memory: bool = False,
    clustering: bool = True,
    force_redo: int = 0,
) -> None:
    os.makedirs(CLUSTERED_DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    if not in_memory:
        os.makedirs(CUT_DATA_DIR, exist_ok=True)
        os.makedirs(DISCRETE_DATA_DIR, exist_ok=True)

    create_one_time_images(region, DISCRETE_DATA_DIR)
    extract_day(dates, region, clean_level, in_memory, clustering, force_redo)
