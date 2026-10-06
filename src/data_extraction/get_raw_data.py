import logging
import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr
from ecmwf.datastores import Client

from . import CUT_DATA_DIR, Region

logger = logging.getLogger(__name__)

logging.getLogger("legacy_client").setLevel(logging.WARNING)
logging.getLogger("ecmwf.datastores").setLevel(logging.WARNING)
logging.getLogger("cdsapi").setLevel(logging.WARNING)


def to_compressed_netcdf(ds: xr.Dataset, path: Path) -> None:
    encoding = {
        name: {
            "zlib": True,
            "complevel": 1,
            "chunksizes": (1,) * (var.ndim - 2) + var.shape[-2:],
        }
        for name, var in ds.data_vars.items()
        if var.ndim >= 2 and var.dtype.kind in "fiu"
    }
    ds.to_netcdf(path, encoding=encoding)


def cut_grib_long_lat(ds: xr.Dataset, coordinates: list[float]) -> xr.Dataset:
    lon_min, lon_max, lat_min, lat_max = coordinates
    lat_min, lat_max = min(lat_min, lat_max), max(lat_min, lat_max)

    lon = ds.longitude
    if lon_min < 0 or lon_max < 0:
        lon = xr.where(lon > 180, lon - 360, lon)

    mask = (
        (lon >= lon_min)
        & (lon <= lon_max)
        & (ds.latitude >= lat_min)
        & (ds.latitude <= lat_max)
    )

    mask_np = mask.values
    y_indices, x_indices = np.where(mask_np)

    y_min = max(0, int(y_indices.min()) - 1)
    y_max = min(ds.sizes["y"], int(y_indices.max()) + 2)
    x_min = max(0, int(x_indices.min()) - 1)
    x_max = min(ds.sizes["x"], int(x_indices.max()) + 2)

    ds_sub = ds.isel(
        y=slice(y_min, y_max),
        x=slice(x_min, x_max),
    )

    if lon_min < 0 or lon_max < 0:
        ds_sub = ds_sub.assign_coords(
            longitude=xr.where(
                ds_sub.longitude > 180, ds_sub.longitude - 360, ds_sub.longitude
            )
        )

    return ds_sub


def cut_path(target_date: date, region: Region, base_path: Path) -> Path:
    day = target_date.strftime("%Y-%m-%d")
    return base_path / CUT_DATA_DIR / day / f"{day}_{region.name}_cut.nc"


def cut_month(
    month_dates: list[date],
    region: Region,
    raw_dir: Path,
    base_path: Path,
    force_redo: bool,
    delete_grib: bool,
) -> list[date]:
    todo = [
        d
        for d in month_dates
        if force_redo or not cut_path(d, region, base_path).exists()
    ]
    if not todo:
        logger.debug(f"ALREADY CUT: {month_dates[0].strftime('%Y-%m')}")
        return month_dates

    month = todo[0].strftime("%Y-%m")
    month_dir = raw_dir / month
    grib_path = month_dir / f"{month}.grib"
    try:
        if grib_has_days(grib_path, todo):
            logger.debug(f"GRIB already exists: {grib_path}")
        else:
            shutil.rmtree(month_dir, ignore_errors=True)
            month_dir.mkdir(parents=True, exist_ok=True)
            download_month_grib(todo, grib_path)
        cut_grib_days(grib_path, todo, region, base_path)
    finally:
        if delete_grib:
            shutil.rmtree(month_dir, ignore_errors=True)

    return [d for d in month_dates if cut_path(d, region, base_path).exists()]


def grib_has_days(grib_path: Path, dates: list[date]) -> bool:
    if not grib_path.exists():
        return False
    try:
        with xr.open_dataset(grib_path, engine="cfgrib") as ds:
            available = set(np.atleast_1d(ds.time.dt.date.values))
    except Exception:
        logger.warning(f"Unreadable GRIB, re-downloading: {grib_path}", exc_info=True)
        return False
    return set(dates) <= available


def cut_grib_days(
    grib_path: Path, dates: list[date], region: Region, base_path: Path
) -> None:
    for d in dates:
        output_path = cut_path(d, region, base_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"CUTTING GRIB: {grib_path} -> {output_path}")
        with xr.open_dataset(
            grib_path,
            engine="cfgrib",
            decode_cf=True,
            decode_times=True,
            backend_kwargs={"filter_by_keys": {"dataDate": int(f"{d:%Y%m%d}")}},
        ) as ds:
            to_compressed_netcdf(cut_grib_long_lat(ds, region.value), output_path)


def download_month_grib(dates: list[date], grib_path: Path) -> None:
    month = dates[0].strftime("%Y-%m")
    logger.debug(f"Downloading GRIB for {month} ({len(dates)} days) to {grib_path}...")
    progress = logger.getEffectiveLevel() == logging.DEBUG
    client = Client(
        key=os.getenv("ECMWF_API_KEY", None),
        url=os.getenv("ECMWF_API_URL", None),
        progress=progress,
    )
    if not client.check_authentication():
        logger.critical("Failed to authenticate with ECMWF API.")
        raise RuntimeError("Failed to authenticate with ECMWF API.")

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
        ],
        "data_format": "grib",
    }

    try:
        client.retrieve(
            "reanalysis-cerra-pressure-levels",
            {
                **base_request,
                "year": [dates[0].strftime("%Y")],
                "month": [dates[0].strftime("%m")],
                "day": [d.strftime("%d") for d in dates],
            },
            str(grib_path),
        )
        logger.debug(f"Download complete: {grib_path}")
    except Exception as e:
        logger.error(f"Failed to download GRIB for {month}: {e}")
        raise
