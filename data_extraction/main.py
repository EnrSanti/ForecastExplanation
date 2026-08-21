import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List

from . import Region, RAW_DATA_DIR, CUT_DATA_DIR, DISCRETE_DATA_DIR, CLUSTERED_DATA_DIR
from .extract_features_nc import create_one_time_images, save_feature_maps, render_feature_maps
from .get_raw_data import extract_nc, extract_nc_in_memory
from .image_proc import cluster, cluster_in_memory

logger = logging.getLogger(__name__)


def extract_day_worker(date, region, clean_level: int = 0, clustering: bool = True):
    clustered_dir = os.path.join(CLUSTERED_DATA_DIR, date.strftime("%Y-%m-%d"))
    raw_data_dir = os.path.join(RAW_DATA_DIR, date.strftime("%Y-%m-%d"))
    cut_data_dir = os.path.join(CUT_DATA_DIR, date.strftime("%Y-%m-%d"))
    discrete_data_dir = os.path.join(DISCRETE_DATA_DIR, date.strftime("%Y-%m-%d"))

    if os.path.exists(clustered_dir) and len(os.listdir(clustered_dir)) == 0:  # todo: flag to force re-clustering
        logger.info(f"Output folder '{clustered_dir}' already contains images. Use flag to force re-clustering.")
        return

    os.makedirs(clustered_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(cut_data_dir, exist_ok=True)
    os.makedirs(discrete_data_dir, exist_ok=True)

    if len(os.listdir(discrete_data_dir)) == 0:
        nc_file = extract_nc(date, region, raw_data_dir, cut_data_dir)
        if not nc_file:
            return
        save_feature_maps(nc_file, region, discrete_data_dir)
    else:
        logger.info(f"Feature maps already exist for {date.strftime('%Y-%m-%d')}, skipping.")

    if clustering:
        cluster(discrete_data_dir, clustered_dir, CLUSTERED_DATA_DIR)

    if clean_level >= 1:
        shutil.rmtree(raw_data_dir, ignore_errors=True)
        shutil.rmtree(cut_data_dir, ignore_errors=True)
    if clean_level >= 2:
        shutil.rmtree(discrete_data_dir, ignore_errors=True)


def extract_day_worker_in_memory(date, region, clean_level: int = 0, clustering: bool = True):
    """In-memory pipeline: GRIB -> xr.Dataset -> Dict[images] -> cluster -> disk output."""
    clustered_dir = os.path.join(CLUSTERED_DATA_DIR, date.strftime("%Y-%m-%d"))
    raw_data_dir = os.path.join(RAW_DATA_DIR, date.strftime("%Y-%m-%d"))

    os.makedirs(clustered_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)

    ds = extract_nc_in_memory(date, region, raw_data_dir)
    images = render_feature_maps(ds, region)
    ds.close()
    if clustering:
        cluster_in_memory(images, clustered_dir, CLUSTERED_DATA_DIR)

    if clean_level >= 1:
        shutil.rmtree(raw_data_dir, ignore_errors=True)


def extract_day(dates: List[datetime], region: Region, clean_level: int = 0, in_memory: bool = False, clustering: bool = True) -> None:
    logger.info("Starting data extraction...")

    worker = extract_day_worker_in_memory if in_memory else extract_day_worker

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(worker, date, region, clean_level, clustering): date
            for date in dates
        }

        for future in as_completed(futures):
            date = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Extract failed for {date}", exc_info=True)

    logger.info("Data extraction completed.")


def extract(dates: List[datetime], region: Region, clean_level: int = 0, in_memory: bool = False, clustering: bool = True) -> None:
    os.makedirs(CLUSTERED_DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    if not in_memory:
        os.makedirs(CUT_DATA_DIR, exist_ok=True)
        os.makedirs(DISCRETE_DATA_DIR, exist_ok=True)

    create_one_time_images(region, CLUSTERED_DATA_DIR)
    extract_day(dates, region, clean_level, in_memory, clustering)
