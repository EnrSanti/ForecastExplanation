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
    TEMPERATURE = {"min_blob_size": 100, "target": "upper", "smooth": 8}
    HUMIDITY = {"min_blob_size": 100, "target": "upper", "smooth": 8}
    CLOUDS = {"min_blob_size": 100, "target": "upper", "smooth": 8}
    WIND = {"min_blob_size": 100, "target": "upper", "smooth": 8}


from .main import run_tobac