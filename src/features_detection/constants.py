from enum import Enum

LEVELS = [1000, 925, 850, 700, 500, 300]
FOLDERS_HEIGHT_SUFF = [f"_at_{l:04d}m" for l in LEVELS]

DEFAULT_V_MAX_AT_HEIGHT = {
    "_at_0100m": 20,
    "_at_0750m": 25,
    "_at_1400m": 25,
    "_at_3000m": 35,
    "_at_5500m": 45,
    "_at_9000m": 70,
}
DEFAULT_DXY = 2500
DEFAULT_DT = 3600
# DEFAULT_V_MAX = 70 #deprecated it changes with the levels
DEFAULT_GAP_FRAMES = 1
DEFAULT_MIN_DISTANCE = 1000
DEFAULT_SMOOTH = 2
DEFAULT_BORDER_THICKNESS = 8
DEFAULT_TIME_OFFSET_HOURS = 0


class WeatherPhenomenon(Enum):
    TEMPERATURE = "temp"
    HUMIDITY = "humidity"
    CLOUDS = "cloud"
    WIND = "winds"


# Parameters scaled for native 95x76 grid (from old 800x915 pixel grid)
class WeatherPhenomenonTobacParams(Enum):
    TEMPERATURE = {
        "min_blob_size": 1,
        "target": "maximum",
        "smooth": 2,
        "threshold": 0.5,
        "cmap": "OrRd",
    }
    HUMIDITY = {
        "min_blob_size": 1,
        "target": "minimum",
        "smooth": 2,
        "threshold": 0.5,
        "cmap": "YlGnBu",
    }
    CLOUDS = {
        "min_blob_size": 1,
        "target": "maximum",
        "smooth": 2,
        "threshold": 0.5,
        "cmap": "viridis",
    }
    WIND = {
        "min_blob_size": 1,
        "target": "maximum",
        "smooth": 2,
        "threshold": 0.5,
        "cmap": "viridis",
    }


RAW_FEATURES_VARS = ["wind", "raw_temp"]
