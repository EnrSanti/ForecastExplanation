import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import cdsapi
import xarray as xr

from . import Region, RAW_DATA_DIR, CUT_DATA_DIR
from typing import List, Tuple
from .extract_features_nc import extract_features_from_nc

logger = logging.getLogger(__name__)


def cut_grib_long_lat(grib_path: str, output_path: str, coordinates: Tuple[float, float, float, float]) -> None:
    ds = xr.open_dataset(grib_path, engine="cfgrib", decode_cf=True, decode_times=True, decode_timedelta=False)
    mask = (ds.longitude >= coordinates[0]) & (ds.longitude <= coordinates[1]) & \
           (ds.latitude >= coordinates[2]) & (ds.latitude <= coordinates[3])

    ds_sub = ds.where(mask, drop=True)
    ds_sub.to_netcdf(output_path)

    ds.close()
    ds_sub.close()


def extract_nc_worker(date: datetime, coordinates: Region) -> None:
    base_name = date.strftime("%Y-%m-%d")
    grib_file = f"{base_name}.grib"
    grib_path = os.path.join(RAW_DATA_DIR, grib_file)
    output_path = os.path.join(CUT_DATA_DIR, base_name + "_" + coordinates.name + "_cut.nc")

    if not os.path.exists(output_path):
        download_grib_if_needed(date, grib_path)

        logger.debug(f"CUTTING GRIB: {grib_path} -> {output_path}")
        cut_grib_long_lat(grib_path, output_path, coordinates.value)
    else:
        logger.debug(f"ALREADY CUT: {output_path}")


def extract_nc(dates: List[datetime], region: Region) -> None:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CUT_DATA_DIR, exist_ok=True)

    logger.info("Starting data extraction...")

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(extract_nc_worker, date, region): date
            for date in dates
        }

        for future in as_completed(futures):
            date = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Extract failed for {date}: {e}")

    logger.info("Data extraction completed.")


def extract(dates: List[datetime], region: Region, output_dir: str) -> None:
    extract_nc(dates, region)
    extract_features_from_nc(dates, region, output_dir)


def download_grib_if_needed(date: datetime, grib_path: str) -> None:
    if os.path.exists(grib_path):
        logger.debug(f"GRIB already exists: {grib_path}")
        return

    date_str = date.strftime("%Y-%m-%d")
    logger.debug(f"Downloading GRIB for {date_str} to {grib_path}...")
    client = cdsapi.Client()
    year, month, day = date_str.split("-")

    base_request = {
        "variable": [
            "cloud_cover",
            "relative_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "pressure_level": [
            "300", "500", "700", "850", "925", "1000"
        ],
        "data_type": ["reanalysis"],
        "product_type": ["forecast"],
        "time": [
            "00:00", "03:00", "06:00", "09:00",
            "12:00", "15:00", "18:00", "21:00"
        ],
        "leadtime_hour": [
            "1", "2", "3", "4", "5", "6", "9", "12", "15", "18", "21", "24", "27"
        ],
        "data_format": "grib"
    }

    try:
        client.retrieve(
            "reanalysis-cerra-pressure-levels",
            {
                **base_request,
                "year": [year],
                "month": [month],
                "day": [day]
            },
            grib_path
        )
        logger.debug(f"Download complete: {grib_path}")
    except Exception as e:
        logger.error(f"Failed to download GRIB for {date_str}: {e}")
        raise
