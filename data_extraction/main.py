import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List


from . import Region, RAW_DATA_DIR, CUT_DATA_DIR, DISCRETE_DATA_DIR
from .extract_features_nc import create_one_time_images, save_feature_maps
from .image_proc import cluster
from .get_raw_data import extract_nc

logger = logging.getLogger(__name__)


def extract_day_worker(date,region, root_output_dir):

    output_dir = os.path.join(root_output_dir, date.strftime("%Y-%m-%d"))
    raw_data_dir = os.path.join(RAW_DATA_DIR, date.strftime("%Y-%m-%d"))
    cut_data_dir = os.path.join(CUT_DATA_DIR, date.strftime("%Y-%m-%d"))
    discrete_data_dir = os.path.join(DISCRETE_DATA_DIR, date.strftime("%Y-%m-%d"))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(cut_data_dir, exist_ok=True)
    os.makedirs(discrete_data_dir, exist_ok=True)
    
    nc_file = extract_nc(date,region, raw_data_dir, cut_data_dir)
    if not nc_file:
        return

    if len(os.listdir(discrete_data_dir)) == 0:
        save_feature_maps(nc_file,region, discrete_data_dir)
    else:
        logger.info(f"Feature maps already exist for {date.strftime('%Y-%m-%d')}, skipping.")

    if len(os.listdir(output_dir)) == 0 or True:  # todo: flag to force re-clustering
        cluster(discrete_data_dir, output_dir, root_output_dir)
    else:
        logger.info(f"Output folder '{output_dir}' already contains images. Skipping clustering.")

    # delete if needed
    # pathlib.Path(os.path.join(RAW_DATA_DIR, date.strftime("%Y-%m-%d"))).rmdir()
    # pathlib.Path(os.path.join(CUT_DATA_DIR, date.strftime("%Y-%m-%d"))).rmdir()
    # pathlib.Path(os.path.join(DISCRETE_DATA_DIR, date.strftime("%Y-%m-%d"))).rmdir()

def extract_day(dates: List[datetime], region: Region, output_dir) -> None:
    logger.info("Starting data extraction...")

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(extract_day_worker, date, region, output_dir): date
            for date in dates
        }

        for future in as_completed(futures):
            date = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Extract failed for {date}", exc_info=True)

    logger.info("Data extraction completed.")


def extract(dates: List[datetime], region: Region, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CUT_DATA_DIR, exist_ok=True)
    os.makedirs(DISCRETE_DATA_DIR, exist_ok=True)


    create_one_time_images(region, output_dir)
    extract_day(dates, region, output_dir)