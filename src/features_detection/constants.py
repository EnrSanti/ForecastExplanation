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
DEFAULT_BORDER_THICKNESS = 8
DEFAULT_TIME_OFFSET_HOURS = 0


class WeatherPhenomenon(Enum):
    TEMPERATURE = "temp"
    HUMIDITY = "humidity"
    CLOUDS = "cloud"
    WIND = "winds"
    FRONT = "front"
    WARM_ADVECTION = "warmadv"
    THETA_E_FALL = "tefall"
    THETA_E_RISE = "terise"


# Parameters scaled for native 95x76 grid (from old 800x915 pixel grid)
FRONT_LEVELS_SUFF = ["_at_0850m", "_at_0700m", "_at_0500m"]


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
    WIND = {  # noqa: RUF012 PIE796
        "min_blob_size": 1,
        "target": "maximum",
        "smooth": 2,
        "threshold": 0.5,
        "cmap": "viridis",
    }

    FRONT = {  # noqa: RUF012
        "levels": FRONT_LEVELS_SUFF,
        "min_blob_size": 20,
        "target": "maximum",
        "smooth": 2,
        "threshold": 0.5,
        "cmap": "magma",
    }
    WARM_ADVECTION = {  # noqa: RUF012
        "field": "tadv",
        "levels": FRONT_LEVELS_SUFF,
        "min_blob_size": 20,
        "target": "maximum",
        "smooth": 2,
        "threshold": 0.6,
        "cmap": "RdBu_r",
    }
    THETA_E_FALL = {  # noqa: RUF012
        "field": "te_change",
        "levels": FRONT_LEVELS_SUFF,
        "min_blob_size": 20,
        "target": "minimum",
        "smooth": 2,
        "threshold": 0.35,
        "cmap": "RdBu_r",
    }
    THETA_E_RISE = {  # noqa: RUF012
        "field": "te_change",
        "levels": FRONT_LEVELS_SUFF,
        "min_blob_size": 20,
        "target": "maximum",
        "smooth": 2,
        "threshold": 0.65,
        "cmap": "RdBu_r",
    }


FRONT_PHENOMENA = [
    WeatherPhenomenon.FRONT,
    WeatherPhenomenon.WARM_ADVECTION,
    WeatherPhenomenon.THETA_E_FALL,
    WeatherPhenomenon.THETA_E_RISE,
]


RAW_FEATURES_VARS = [
    "wind",
    "raw_temp",
    "raw_humidity",
    "raw_front",
    "raw_tadv",
    "raw_te_change",
]
