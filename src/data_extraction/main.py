import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List
import cv2
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

from . import (
    Region,
    RAW_DATA_DIR,
    CUT_DATA_DIR,
    DISCRETE_DATA_DIR,
    CLUSTERED_DATA_DIR,
    FOLDERS,
)
from .extract_features_nc import (
    create_one_time_images,
    build_feature_dataarrays,
)
from .get_raw_data import extract_nc
from .clustering import cluster_xarray

logger = logging.getLogger(__name__)


def find_starting_step(
    clustered_dir: str, raw_data_dir: str, cut_data_dir: str, discrete_data_dir: str
) -> int:
    """Find the starting step for the data extraction process."""
    if os.path.exists(clustered_dir) and len(os.listdir(clustered_dir)) > 0:
        return 4  # Clustering done
    if os.path.exists(discrete_data_dir) and len(os.listdir(discrete_data_dir)) > 0:
        return 3  # Feature maps saved
    if os.path.exists(cut_data_dir) and len(os.listdir(cut_data_dir)) > 0:
        return 2  # GRIB cut
    if os.path.exists(raw_data_dir) and len(os.listdir(raw_data_dir)) > 0:
        return 1  # GRIB downloaded

    return 0  # No data


def extract_day_worker(
    date,
    region,
    base_path: str,
    clean_level: int = 0,
    clustering: bool = True,
    force_redo: bool = False,
    just_cut: bool = False,
    create_images: bool = False,
):
    logger.debug(f"Extracting data for {date.strftime('%Y-%m-%d')}")
    clustered_dir = os.path.join(
        base_path, CLUSTERED_DATA_DIR, date.strftime("%Y-%m-%d")
    )
    # raw data can be shared between runs
    raw_data_dir = os.path.join(RAW_DATA_DIR, date.strftime("%Y-%m-%d"))
    cut_data_dir = os.path.join(base_path, CUT_DATA_DIR, date.strftime("%Y-%m-%d"))
    discrete_data_dir = os.path.join(
        base_path, DISCRETE_DATA_DIR, date.strftime("%Y-%m-%d")
    )
    features_nc_path = os.path.join(discrete_data_dir, "features.nc")

    starting_step = find_starting_step(
        clustered_dir, raw_data_dir, cut_data_dir, discrete_data_dir
    )
    nc_file = ""

    if starting_step in [0, 1] or force_redo:
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(cut_data_dir, exist_ok=True)
        # this skips 0 1 automatically if already done
        nc_file = extract_nc(date, region, raw_data_dir, cut_data_dir, force_redo)
        starting_step = 2

    if (starting_step == 2 or force_redo) and not just_cut:
        os.makedirs(discrete_data_dir, exist_ok=True)
        feature_data = build_feature_dataarrays(nc_file)
        feature_data.to_netcdf(features_nc_path)
        if create_images:
            save_tobac_input_images(feature_data, discrete_data_dir)
        starting_step = 3

    if ((starting_step == 3 or force_redo) and clustering) and not just_cut:
        os.makedirs(clustered_dir, exist_ok=True)
        with xr.open_dataset(features_nc_path) as features_ds:
            feature_data = {str(name): da for name, da in features_ds.data_vars.items()}
        clustered_data = cluster_xarray(feature_data)
        clustered_data.to_netcdf(os.path.join(clustered_dir, "features.nc"))

    if starting_step == 4:
        logger.debug(
            f"Feature maps already exist for {date.strftime('%Y-%m-%d')}, skipping."
        )

    if clean_level >= 1:
        shutil.rmtree(raw_data_dir, ignore_errors=True)
    if clean_level >= 2:
        shutil.rmtree(cut_data_dir, ignore_errors=True)
    if clean_level >= 3:
        shutil.rmtree(discrete_data_dir, ignore_errors=True)
        shutil.rmtree(clustered_dir, ignore_errors=True)


def save_tobac_input_images(feature_data: xr.Dataset, output_dir: str) -> None:
    """
    Renders each (variable, level, time) slice in feature_data to a raw
    grayscale PNG — written directly from normalized pixel values, not
    through a matplotlib colormap, since downstream code (convert_frames_to_
    grayscale) just converts back to grayscale anyway; skipping the color
    round-trip avoids the precision loss that introduces.

    Layout matches what the rest of the pipeline expects to read back in
    (FOLDERS_HEIGHT_SUFF, extract_keys/extract_times):
        output_dir/<variable><level_suffix>/<variable>_<YYYYMMDD_HHMM>.png

    Assumes feature_data has dims (time, level, y, x) with level values
    matching LEVEL_TO_SUFFIX's keys — adjust the dim name / mapping if
    build_feature_dataarrays uses something different.

    Normalization is per (variable, level), computed across all of that
    variable+level's time steps for the day — not per individual frame — so
    a flat/no-signal frame comes out uniformly dark against the day's real
    range instead of being stretched to fill [0, 255] on its own and looking
    spuriously bright (the cause of the earlier "whole image reads as one
    cloud" bug).
    """
    os.makedirs(output_dir, exist_ok=True)

    for var_name in feature_data.data_vars:
        da = feature_data[var_name]

        has_level = "level" in da.dims
        levels = da["level"].values if has_level else [None]

        for level in levels:
            level_da = da.sel(level=level) if has_level else da
            suffix = (
                FOLDERS.get(int(level), f"_at_{level}") if level is not None else ""
            )
            var_dir = os.path.join(output_dir, f"{var_name}{suffix}")
            os.makedirs(var_dir, exist_ok=True)

            vmin = float(level_da.min())
            vmax = float(level_da.max())
            vrange = (
                vmax - vmin if vmax > vmin else 1.0
            )  # guard against a fully-flat day

            for t in range(level_da.sizes["time"]):
                frame = level_da.isel(time=t)
                ts = pd.to_datetime(frame["time"].values)

                norm = (
                    ((frame.values - vmin) / vrange * 255).clip(0, 255).astype(np.uint8)
                )

                fname = f"{var_name}_{ts.strftime('%Y%m%d_%H%M')}.png"
                cv2.imwrite(os.path.join(var_dir, fname), norm)

    logger.info(f"Saved TOBAC input images to '{output_dir}'.")


def extract_day(
    dates: List[datetime],
    region: Region,
    base_path: str,
    clean_level: int = 0,
    clustering: bool = True,
    force_redo: bool = False,
    just_cut: bool = False,
    create_images: bool = False,
) -> None:
    logger.info("Starting data extraction...")

    worker = extract_day_worker

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(
                worker,
                date,
                region,
                base_path,
                clean_level,
                clustering,
                force_redo,
                just_cut,
                create_images,
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
    output_path: str,
    clean_level: int = 0,
    clustering: bool = True,
    force_redo: bool = False,
    just_cut: bool = False,
    create_images: bool = False,
) -> None:

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, CLUSTERED_DATA_DIR), exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(output_path, CUT_DATA_DIR), exist_ok=True)
    os.makedirs(os.path.join(output_path, DISCRETE_DATA_DIR), exist_ok=True)
    os.makedirs(os.path.join(output_path, "legends"), exist_ok=True)

    if create_images:
        create_one_time_images(region, os.path.join(output_path, "legends"))

    extract_day(
        dates,
        region,
        output_path,
        clean_level,
        clustering,
        force_redo,
        just_cut,
        create_images=create_images,
    )
