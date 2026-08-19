from enum import Enum


class Region(Enum):
    FVG = [11, 15, 44.5, 48]
    ITALY = [10, 16, 42, 48]


RAW_DATA_DIR = "./raw_data/data/original_CERRA"
CUT_DATA_DIR = "./raw_data/data/CERRA_cut"

from .get_raw_data import extract

__all__ = [
    "extract",
    "Region",
]
