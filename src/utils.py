import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

LOG_FORMAT = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
PROJECT_LOGGERS = ("ForecastExplanation", "data_extraction", "features_detection")

LEVELS = [1000, 925, 850, 700, 500, 300]
FOLDERS = {lev: f"_at_{lev:04d}m" for lev in LEVELS}

DATE_COLUMN = "date"


def setup_logging(level: int) -> None:
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
    for name in PROJECT_LOGGERS:
        logging.getLogger(name).setLevel(level)


def worker_pool(workers: int) -> ProcessPoolExecutor:
    level = logging.getLogger(PROJECT_LOGGERS[0]).getEffectiveLevel()
    return ProcessPoolExecutor(
        max_workers=workers, initializer=setup_logging, initargs=(level,)
    )


def to_compressed_netcdf(ds: xr.Dataset, path: Path) -> None:
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name, var in ds.data_vars.items()
        if var.dtype.kind in "fiu"
    }
    ds.to_netcdf(path, encoding=encoding)


def haversine(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c
