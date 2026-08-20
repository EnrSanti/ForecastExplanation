from enum import Enum


class Region(Enum):
    FVG = [11, 15, 44.5, 48]
    ITALY = [10, 16, 42, 48]


RAW_DATA_DIR = "./tmp_data/original_CERRA"
CUT_DATA_DIR = "./tmp_data/CERRA_cut"
DISCRETE_DATA_DIR = "./tmp_data/imgs_discrete"

from .main import extract

__all__ = [
    "extract",
    "Region",
    "RAW_DATA_DIR",
    "CUT_DATA_DIR",
    "DISCRETE_DATA_DIR"
]
