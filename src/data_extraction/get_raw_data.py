import logging
import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr
from ecmwf.datastores import Client

from region import Region
from utils import to_compressed_netcdf

from . import CUT_DATA_DIR

logger = logging.getLogger(__name__)

logging.getLogger("ecmwf.datastores").setLevel(logging.WARNING)


def cut_grib_long_lat(ds: xr.Dataset, coordinates: list[float]) -> xr.Dataset:
    lon_min, lon_max, lat_min, lat_max = coordinates

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

    ds_sub = ds.isel(y=slice(y_min, y_max), x=slice(x_min, x_max))

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
    month = month_dates[0].strftime("%Y-%m")
    month_dir = raw_dir / month
    grib_path = month_dir / f"{month}.grib"
    todo = [
        d
        for d in month_dates
        if force_redo or not cut_path(d, region, base_path).exists()
    ]
    if todo:
        available = grib_days(grib_path)
        missing = [d for d in todo if d not in available]
        if missing:
            add_to_grib(missing, grib_path, keep=bool(available))
        else:
            logger.debug(f"GRIB already exists: {grib_path}")
        cut_grib_days(grib_path, todo, region, base_path)
    else:
        logger.debug(f"ALREADY CUT: {month}")

    # only reached once every day is cut: a failed download or cut keeps the GRIB
    if delete_grib:
        shutil.rmtree(month_dir, ignore_errors=True)
    return [d for d in month_dates if cut_path(d, region, base_path).exists()]


def grib_days(grib_path: Path) -> set[date]:
    if not grib_path.exists():
        return set()
    try:
        with xr.open_dataset(grib_path, engine="cfgrib") as ds:
            return set(np.atleast_1d(ds.time.dt.date.values))
    except Exception:
        logger.warning(f"Unreadable GRIB, re-downloading: {grib_path}", exc_info=True)
        return set()


def add_to_grib(dates: list[date], grib_path: Path, keep: bool) -> None:
    part_path = grib_path.with_name(grib_path.name + ".part")
    grib_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_month_grib(dates, part_path)
        if keep:
            with open(grib_path, "ab") as grib, open(part_path, "rb") as part:
                shutil.copyfileobj(part, grib)
        else:
            part_path.replace(grib_path)
    finally:
        part_path.unlink(missing_ok=True)
    # cfgrib index of the old content
    for index in grib_path.parent.glob(f"{grib_path.name}.*.idx"):
        index.unlink()


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
        key=os.getenv("ECMWF_API_KEY"),
        url=os.getenv("ECMWF_API_URL"),
        progress=progress,
    )
    if not client.check_authentication():
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
        "leadtime_hour": ["1", "2", "3"],
        "data_format": "grib",
    }

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
