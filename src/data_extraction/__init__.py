from typing import ClassVar

from region import Region


class LimitValues:
    CLOUD: ClassVar[dict[int, tuple[int, int]]] = {
        1000: (0, 100),
        925: (0, 100),
        850: (0, 100),
        700: (0, 100),
        500: (0, 100),
        300: (0, 100),
    }

    # One temperature level per layer
    TEMP: ClassVar[dict[int, tuple[float, float]]] = {
        1000: (263.15, 311.15),
        925: (259.15, 306.15),
        850: (254.15, 302.15),
        700: (244.15, 292.15),
        500: (230.15, 281.15),
        300: (218.15, 268.15),
    }

    WIND_SPEED: ClassVar[dict[int, tuple[int, int]]] = {
        1000: (0, 100),
        925: (0, 100),
        850: (0, 100),
        700: (0, 120),
        500: (0, 150),
        300: (0, 200),
    }

    HUMIDITY: ClassVar[dict[int, tuple[int, int]]] = {
        1000: (0, 100),
        925: (0, 100),
        850: (0, 100),
        700: (0, 100),
        500: (0, 100),
        300: (0, 100),
    }


RAW_DATA_DIR: str = "tmp_data/original_CERRA"
CUT_DATA_DIR: str = "tmp_data/CERRA_cut"
DISCRETE_DATA_DIR: str = "tmp_data/imgs_discrete"
CLUSTERED_DATA_DIR: str = "tmp_data/clustered"

LEVELS = [1000, 925, 850, 700, 500, 300]
FOLDERS = {l: f"_at_{l:04d}m" for l in LEVELS}

from .main import extract

__all__: list[str] = [
    "CLUSTERED_DATA_DIR",
    "CUT_DATA_DIR",
    "DISCRETE_DATA_DIR",
    "RAW_DATA_DIR",
    "LimitValues",
    "Region",
    "extract",
]
