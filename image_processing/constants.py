from enum import Enum

TOBAC_OUTPUT = "./data/tracked"

FOLDERS_HEIGHT_SUFF = ["_at_100m", "_at_750m", "_at_1_4km", "_at_3km", "_at_5_5km", "_at_9km"]

DEFAULT_DXY = 2500
DEFAULT_DT = 3600
DEFAULT_V_MAX = 70
DEFAULT_GAP_FRAMES = 1
DEFAULT_MIN_DISTANCE = 1000
DEFAULT_SMOOTH = 8
DEFAULT_BORDER_THICKNESS = 8
DEFAULT_TIME_OFFSET_HOURS = 0


class Region(Enum):
    FVG = [11, 15, 44.5, 48]
    ITALY = [10, 16, 42, 48]


class WeatherPhenomenon(Enum):
    TEMPERATURE = "temp"
    HUMIDITY = "humidity"
    CLOUDS = "cloud"
    WIND = "winds"


class WeatherPhenomenonTobacParams(Enum):
    TEMPERATURE = {"min_blob_size": 100, "target": "maximum", "smooth": 8, "threshold": 0.6 , "cmap": "OrRd"}
    HUMIDITY = {"min_blob_size": 100, "target": "minimum", "smooth": 8, "threshold": 0.55, "cmap": "YlGnBu"}
    CLOUDS = {"min_blob_size": 100, "target": "maximum", "smooth": 8, "threshold": 0.7, "cmap": "viridis"}
    WIND = {"min_blob_size": 100, "target": "maximum", "smooth": 8, "threshold": 0.6, "cmap": "viridis"}

