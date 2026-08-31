import logging
import os
from datetime import datetime
from typing import List, Optional

import cdsapi
import xarray as xr

from . import Region

import logging

logging.basicConfig(level=logging.INFO)  # your existing setup, presumably

logger = logging.getLogger(__name__)  # your own logger — stays at INFO

# silence the noisy third-party library specifically
logging.getLogger("legacy_client").setLevel(logging.WARNING)
logging.getLogger("ecmwf.datastores").setLevel(logging.WARNING)
logging.getLogger("cdsapi").setLevel(logging.WARNING)


def cut_grib_long_lat(grib_path: str, coordinates: List[int]) -> Optional[xr.Dataset]:
    with xr.open_dataset(
        grib_path,
        engine="cfgrib",
        decode_cf=True,
        decode_times=True,
        decode_timedelta=False,
    ) as ds:
        mask = (
            (ds.longitude >= coordinates[0] - 0.5)
            & (ds.longitude <= coordinates[1] + 0.5)
            & (ds.latitude >= coordinates[2] - 0.5)
            & (ds.latitude <= coordinates[3] + 0.5)
        )
    with xr.open_dataset(
        grib_path,
        engine="cfgrib",
        decode_cf=True,
        decode_times=True,
        decode_timedelta=False,
    ) as ds:
        mask = (
            (ds.longitude >= coordinates[0] - 0.5)
            & (ds.longitude <= coordinates[1] + 0.5)
            & (ds.latitude >= coordinates[2] - 0.5)
            & (ds.latitude <= coordinates[3] + 0.5)
        )

        ds_sub = ds.where(mask, drop=True)
        return ds_sub


def extract_nc(
    date: datetime, region: Region, input_dir: str, output_dir: str, force_redo: int
) -> str:
    base_name = date.strftime("%Y-%m-%d")
    grib_file = f"{base_name}.grib"
    grib_path = os.path.join(input_dir, grib_file)
    output_path = os.path.join(output_dir, base_name + "_" + region.name + "_cut.nc")

    if not os.path.exists(output_path) or force_redo >= 3:
        download_grib_if_needed(date, grib_path)

        logger.debug(f"CUTTING GRIB: {grib_path} -> {output_path}")
        ds = cut_grib_long_lat(grib_path, region.value)
        ds.to_netcdf(output_path)
        ds.close()

    else:
        logger.debug(f"ALREADY CUT: {output_path}")

    return output_path


def extract_nc_in_memory(
    date: datetime, region: Region, raw_data_dir: str
) -> xr.Dataset:
    """Download GRIB (if needed) and return cropped dataset in memory."""
    base_name = date.strftime("%Y-%m-%d")
    grib_path = os.path.join(raw_data_dir, f"{base_name}.grib")

    download_grib_if_needed(date, grib_path)

    logger.debug(f"CUTTING GRIB (in-memory): {grib_path}")
    return cut_grib_long_lat(grib_path, region.value).load()


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
            "v_component_of_wind",
        ],
        "pressure_level": ["300", "500", "700", "850", "925", "1000"],
        "data_type": ["reanalysis"],
        "product_type": ["forecast"],
        "time": [
            "00:00",
            "03:00",
            "06:00",
            "09:00",
            "12:00",
            "15:00",
            "18:00",
            "21:00",
        ],
        "leadtime_hour": [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "9",
            "12",
            "15",
            "18",
            "21",
            "24",
            "27",
        ],
        "data_format": "grib",
    }

    try:
        client.retrieve(
            "reanalysis-cerra-pressure-levels",
            {**base_request, "year": [year], "month": [month], "day": [day]},
            grib_path,
        )
        logger.debug(f"Download complete: {grib_path}")
    except Exception as e:
        logger.error(f"Failed to download GRIB for {date_str}: {e}")
        raise
