#!/usr/bin/env python3
"""
multi_folder_open_fronts.py

Detect a single open front curve per image — not a closed loop.
Processes multiple input folders, saving results to a parallel folder tree
under a specified OUTPUT_PARENT_DIR.
"""

import os
import cv2
import numpy as np
from PIL import Image

# --- CONFIG ---
INPUT_FOLDERS = [
    "./image_processing/fvg/resized/humidity_at_1_4km/",
    "./image_processing/fvg/resized/humidity_at_3km/",
    "./image_processing/fvg/resized/humidity_at_5_5km/",
    "./image_processing/fvg/resized/humidity_at_9km/",
    "./image_processing/fvg/resized/humidity_at_100m/",
    "./image_processing/fvg/resized/humidity_at_750m/"
]

# All results will be saved under this parent folder,
# preserving the same folder names as INPUT_FOLDERS
OUTPUT_PARENT_DIR = "./image_processing/fvg/output/"

GAUSSIAN_SIGMA_MAIN = 8.0
PERCENTILE = 90
MIN_FRONT_FRACTION = 0.5     # fraction of image diagonal
APPROX_EPSILON = 20.0
LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 3
# ----------------------------




def get_humidity_front(input_folder,folders_suff, output_parent_dir,clustered_or_not):
    pass