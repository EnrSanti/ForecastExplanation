from enum import Enum


class Region(Enum):
    FVG = [11, 15, 44.5, 48]
    ITALY = [10, 16, 42, 48]


class LimitValues:
    MINIMUM_CLOUD_VALUE = 0
    MAXIMUM_CLOUD_VALUE = 100
    MINIMUM_TEMP_VALUE = 215
    MAXIMUM_TEMP_VALUE = 315
    MINIMUM_WIND_SPEED_VALUE = 0
    MAXIMUM_WIND_SPEED_VALUE = 100
    MINIMUM_HUMIDITY_VALUE = 0
    MAXIMUM_HUMIDITY_VALUE = 100


RAW_DATA_DIR: str = "./tmp_data/original_CERRA"
CUT_DATA_DIR: str = "./tmp_data/CERRA_cut"
DISCRETE_DATA_DIR: str = "./tmp_data/imgs_discrete"
CLUSTERED_DATA_DIR: str = "./tmp_data/clustered"

from .main import extract

__all__: list[str] = [
    "extract",
    "Region",
    "RAW_DATA_DIR",
    "CUT_DATA_DIR",
    "DISCRETE_DATA_DIR"
]
