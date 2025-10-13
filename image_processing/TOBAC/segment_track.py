import tobac
import imageio
import os
print('using tobac version', str(tobac.__version__))
import tobac.testing
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import re
import seaborn as sns
import pandas as pd
import scipy.ndimage as ndimage
import imageio as images
sns.set_context("talk")

def locate_and_segment():
    pass
def track():
    pass   


def extract_keys(filename):
    # match pattern like cloud_200_YYYYMMDD_HHMM.png
    m = re.search(r'_(\d{8})_(\d+)\.png$', filename)
    if m:
        date = int(m.group(1))
        num = int(m.group(2))
        return (date, num)
    else:
        return (0, 0)


def run_tobac(input_folder, output_folder):
    
    image_files = ([os.path.join(input_folder, f) for f in os.listdir(input_folder)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    images_no=len(image_files)
    
    image_files = sorted(image_files, key=extract_keys)

    frames = [imageio.v2.imread(f) for f in image_files]


    datetimes = []
    for f in image_files:
        basename = os.path.basename(f)
        # Example: "cloud_123_20251008_1200.png"
        # Split by underscore and take the last two parts
        parts = basename.split("_")
        date_str = parts[-2]      # YYYYMMDD
        hour_str = parts[-1].split(".")[0]  # HHHH
        dt_str = date_str + hour_str        # "YYYYMMDDHHHH"
        
        # Convert to pandas datetime
        dt = pd.to_datetime(dt_str, format="%Y%m%d%H%M")  # assuming HHHH is HHMM
        datetimes.append(dt)

    # Convert to array
    datetimes = pd.to_datetime(datetimes)


    #Convert frames to grayscale
    frames_gray = [np.mean(f[:, :, :3], axis=2) if f.ndim==3 else f for f in frames]

    # Stack into 3D array (time, y, x)
    data = np.stack(frames_gray)
    n_time, n_y, n_x = data.shape

    # Spatial coordinates (example: 1 pixel = 1000 m)
    dx = dy = 3000  # adjust to your case
    x = np.arange(n_x) * dx
    y = -np.arange(n_y) * dy

    #Create xarray.DataArray
    test_data = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": datetimes,
            "y": y,
            "x": x
        },
        name="w",
        attrs={"units": "m s-1"}
    )

    #Optional: latitude / longitude (linear approximation)
    lat_min, lat_max = 40.0, 45.0
    lon_min, lon_max = -5.0, 0.0
    lat = np.linspace(lat_min, lat_max, n_y)
    lon = np.linspace(lon_min, lon_max, n_x)
    latitude = np.tile(lat[:, np.newaxis], (1, n_x))
    longitude = np.tile(lon[np.newaxis, :], (n_y, 1))
    test_data = test_data.assign_coords(latitude=(("y","x"), latitude),
                                        longitude=(("y","x"), longitude))
    

    dxy, dt = tobac.get_spacings(test_data,grid_spacing=(1, 1))


    # === GLOBAL NORMALIZATION ===
    vmin = float(test_data.min())
    vmax = float(test_data.max())
    test_data_norm = (test_data - vmin) / (vmax - vmin)

    # === PARAMETERS ===
    smooth = 8

    # Convert original threshold (155) to normalized [0, 1] scale
    norm_threshold = 0.75


    # === FEATURE DETECTION ===
    features = tobac.feature_detection_multithreshold(
        test_data_norm,
        threshold=[norm_threshold],  # single threshold in normalized space
        dxy=dxy,
        target="minimum",
        position_threshold="center",
        sigma_threshold=smooth
    )

    os.makedirs(output_folder, exist_ok=True)  # create folder if it doesn't exist

    # === PLOTTING AND SAVING ===
    
    for i, itime in enumerate(range(0, images_no)):
        original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Smooth the frame
        smoothed_frame = ndimage.gaussian_filter(
            test_data_norm.isel(time=itime).values, sigma=smooth
        )
        temp_da = test_data_norm.isel(time=itime).copy()
        temp_da.data = smoothed_frame

        # consistent color range
        temp_da.plot(ax=ax, vmin=0, vmax=1, cmap="viridis")

        # overlay detections
        f = features[features["frame"] == itime]
        f.plot.scatter(
            x="x",
            y="y",
            s=20,
            ax=ax,
            color="red",
            marker="x",
        )

        ax.set_title(f"timeframe = {itime}")

        # save figure
        out_path = os.path.join(output_folder, f"{original_img_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # close the figure to free memory

    locate_and_segment()
    print("Locating procedure completed")
    track()
    print("Tracking procedure completed")
    print("Segmenting procedure completed")
