from enum import Enum


TOBAC_OUTPUT = "./data/tracked"

FOLDERS_HEIGHT_SUFF = ["_at_100m", "_at_750m", "_at_1_4km", "_at_3km", "_at_5_5km", "_at_9km"]

class Region(Enum):
    FVG = [11, 15, 44.5, 48]
    ITALY = [10, 16, 42, 48]

class WeatherPhenomenon(Enum):
    TEMPERATURE = "temp"
    HUMIDITY = "humidity"
    CLOUDS = "cloud"
    WIND = "winds"

class WeatherPhenomenonTobacPrams(Enum):
    TEMPERATURE = {"min_blob_size": 100, "target": "maximum", "smooth": 8, "threshold": 0.6}
    HUMIDITY = {"min_blob_size": 100, "target": "minimum", "smooth": 8, "threshold": 0.55}
    CLOUDS = {"min_blob_size": 100, "target": "maximum", "smooth": 8, "threshold": 0.7}
    WIND = {"min_blob_size": 100, "target": "maximum", "smooth": 8, "threshold": 0.6}


from .main import run_tobac