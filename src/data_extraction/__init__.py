RAW_DATA_DIR: str = "tmp_data/original_CERRA"
CUT_DATA_DIR: str = "tmp_data/CERRA_cut"
DISCRETE_DATA_DIR: str = "tmp_data/extracted_data"
CLUSTERED_DATA_DIR: str = "tmp_data/clustered_data"

LEVELS = [1000, 925, 850, 700, 500, 300]
FRONT_LEVELS = [850, 700, 500]
FOLDERS = {lev: f"_at_{lev:04d}m" for lev in LEVELS}

from .get_raw_data import to_compressed_netcdf
from .main import clean_artifacts, extract

__all__: list[str] = [
    "CLUSTERED_DATA_DIR",
    "CUT_DATA_DIR",
    "DISCRETE_DATA_DIR",
    "FOLDERS",
    "RAW_DATA_DIR",
    "clean_artifacts",
    "extract",
    "to_compressed_netcdf",
]
