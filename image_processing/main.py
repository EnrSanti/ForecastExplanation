


from datetime import datetime 
from typing import List
import cv2
import imageio
import imageio as images
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import seaborn as sns
from image_processing import Region, WeatherPhenomenon, WeatherPhenomenonTobacPrams, FOLDERS_HEIGHT_SUFF
import tobac
import os

def run_tobac(dates: List[datetime],  input_dir: str, output_dir: str, region: Region):
    #, minumum_size_blob=100, target="upper", save_split_merges=True, smooth=8
    # phenomenon: WeatherPhenomenon, minumum_size_blob, target, save_split_merges=True, smooth=8 
    os.makedirs(output_dir, exist_ok=True)

    for date in dates:
        #run sui diversi tipi di fenomeno atmosferico

        day_input_dir = os.path.join(input_dir, date.strftime("%Y-%m-%d"))
        day_output_dir = os.path.join(output_dir, date.strftime("%Y-%m-%d"))
        os.makedirs(day_output_dir, exist_ok=True)
        run_tobac_single_day(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.TEMPERATURE, WeatherPhenomenonTobacPrams.TEMPERATURE)
        run_tobac_single_day(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.HUMIDITY, WeatherPhenomenonTobacPrams.HUMIDITY)
        run_tobac_single_day(date, day_input_dir, day_output_dir, region, WeatherPhenomenon.CLOUDS, WeatherPhenomenonTobacPrams.CLOUDS)
        #run_tobac_single_day(date, input_dir, output_dir, region, WeatherPhenomenon.WIND, WeatherPhenomenonTobacPrams.WIND)


def run_tobac_single_day(date: datetime, day_input_dir: str, day_output_dir: str, region: Region,phenomenon: WeatherPhenomenon, pheomenonParams: WeatherPhenomenonTobacPrams):

    #loop only on the possible suffixes of the folders (height)
    for suffix in FOLDERS_HEIGHT_SUFF:
        height_input_dir = os.path.join(day_input_dir, f"{phenomenon.value}{suffix}")
        height_output_dir = os.path.join(day_output_dir, f"{phenomenon.value}{suffix}")
        os.makedirs(height_output_dir, exist_ok=True)

        image_files = ([f for f in os.listdir(height_input_dir)
                            if f.lower().endswith((".png"))])
        images_no = len(image_files)
        image_files = sorted(image_files, key=extract_keys)
        frames = [imageio.v2.imread(f) for f in image_files]

        # convert frames to grayscale
        if phenomenon == WeatherPhenomenon.TEMPERATURE:
            frames_gray = [1 - np.mean(f[:, :, :3], axis=2) if f.ndim == 3 else f for f in frames]
        else:
            frames_gray = [np.mean(f[:, :, :3], axis=2) if f.ndim == 3 else f for f in frames]


        #



        print(f"Running {height_input_dir}")

def extract_keys(filename):
    """
    extracts date and number from filename for sorting purposes. 
    
    Parameters
    ----------
    filename: file name string

    Returns
    ----------
    tuple (date as int YYYYMMDD, number as int)
    """

    # match pattern like cloud_200_YYYYMMDD_HHMM.png
    m = re.search(r'_(\d{8})_(\d+)\.png$', filename)
    if m:
        date = int(m.group(1))
        num = int(m.group(2))
        return (date, num)
    else:
        return (0, 0)