# tracking speed limit per pressure level (hPa), with its approximate altitude
V_MAX_AT_LEVEL = {
    1000: 20,  # ~100m
    925: 25,  # ~750m
    850: 25,  # ~1400m
    700: 35,  # ~3000m
    500: 45,  # ~5500m
    300: 70,  # ~9000m
}
DT = 3600  # s between the hourly frames
GAP_FRAMES = 1
MIN_DISTANCE = 1000


# tobac settings per segmented field (min_blob_size in ~5.5 km grid points)
TOBAC_PARAMS = {
    "temp": {
        "min_blob_size": 200,
        "target": "maximum",
        "smooth": 2.0,
        "threshold": 0.7,
        "cmap": "OrRd",
    },
    "humidity": {
        "min_blob_size": 200,
        "target": "minimum",
        "smooth": 2.0,
        "threshold": 0.6,
        "cmap": "YlGnBu",
    },
    "cloud": {
        "min_blob_size": 1,
        "target": "maximum",
        "smooth": 2.0,
        "threshold": 0.5,
        "cmap": "viridis",
    },
}


RAW_FEATURES_VARS = [
    "wind",
    "raw_cloud",
    "raw_temp",
    "raw_humidity",
    "raw_front",
    "raw_tadv",
    "raw_te_change",
]
