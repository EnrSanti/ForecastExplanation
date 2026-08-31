from enum import Enum


class Region(Enum):
    FVG = [11, 15, 44.5, 48]
    ITALY = [10, 16, 42, 48]


#split by level 
class LimitValues:

    CLOUD = {
        1000: (0, 100),
        925: (0, 100),
        850: (0, 100),
        700: (0, 100),
        500: (0, 100),
        300: (0, 100),
    }

    #one temperature level per layer
    TEMP = {
        1000: (263.15, 311.15),
        925: (259.15, 306.15),
        850: (254.15, 302.15),
        700: (244.15, 292.15),
        500: (230.15, 281.15),
        300: (218.15, 268.15),
    }

    WIND_SPEED = {
        1000: (0, 100),
        925: (0, 100),
        850: (0, 100),
        700: (0, 120),
        500: (0, 150),
        300: (0, 200),
    }

    HUMIDITY = {
        1000: (0, 100),
        925: (0, 100),
        850: (0, 100),
        700: (0, 100),
        500: (0, 100),
        300: (0, 100),
    }





RAW_DATA_DIR: str = "./tmp_data/original_CERRA"
CUT_DATA_DIR: str = "./tmp_data/CERRA_cut"
DISCRETE_DATA_DIR: str = "./tmp_data/imgs_discrete"
CLUSTERED_DATA_DIR: str = "./tmp_data/clustered"

from .main import extract

__all__: list[str] = [
    "extract",
    "Region",
    "LimitValues",
    "RAW_DATA_DIR",
    "CUT_DATA_DIR",
    "DISCRETE_DATA_DIR",
    "CLUSTERED_DATA_DIR",
]
