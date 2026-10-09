from enum import Enum

LEVELS = [1000, 925, 850, 700, 500, 300]
FOLDERS_HEIGHT_SUFF = [f"_at_{lev:04d}m" for lev in LEVELS]

# Keyed by pressure-level suffix (matches FOLDERS_HEIGHT_SUFF); the comment
# gives the approximate real-world altitude for each level.
DEFAULT_V_MAX_AT_HEIGHT = {
    "_at_1000m": 20,  # ~100m
    "_at_0925m": 25,  # ~750m
    "_at_0850m": 25,  # ~1400m
    "_at_0700m": 35,  # ~3000m
    "_at_0500m": 45,  # ~5500m
    "_at_0300m": 70,  # ~9000m
}
DEFAULT_DXY = 5500
DEFAULT_DT = 3600
DEFAULT_GAP_FRAMES = 1
DEFAULT_MIN_DISTANCE = 1000
DEFAULT_SMOOTH = 2


class WeatherPhenomenon(Enum):
    TEMPERATURE = "temp"
    HUMIDITY = "humidity"
    CLOUDS = "cloud"


# min_blob_size in grid points (~5.5 km)
class WeatherPhenomenonTobacParams(Enum):
    TEMPERATURE = {  # noqa: RUF012
        "min_blob_size": 200,
        "target": "maximum",
        "smooth": 2,
        "threshold": 0.7,
        "cmap": "OrRd",
    }
    HUMIDITY = {  # noqa: RUF012
        "min_blob_size": 200,
        "target": "minimum",
        "smooth": 2,
        "threshold": 0.6,
        "cmap": "YlGnBu",
    }
    CLOUDS = {  # noqa: RUF012
        "min_blob_size": 1,
        "target": "maximum",
        "smooth": 2,
        "threshold": 0.5,
        "cmap": "viridis",
    }


RAW_FEATURES_VARS = [
    "wind",
    "raw_temp",
    "raw_humidity",
    "raw_front",
    "raw_tadv",
    "raw_te_change",
]


TOBAC_PHENOMENA = [
    WeatherPhenomenon.TEMPERATURE,
    WeatherPhenomenon.HUMIDITY,
    WeatherPhenomenon.CLOUDS,
]
