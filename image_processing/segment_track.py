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

def locate_and_segment(input_folder, output_folder,smooth = 8):
    
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


    # Convert original threshold (155) to normalized [0, 1] scale
    norm_threshold = 0.75


    # === FEATURE DETECTION ===
    features = tobac.feature_detection_multithreshold(
        test_data_norm,
        threshold=[norm_threshold],  # single threshold in normalized space
        dxy=dxy,
        target="minimum",
        position_threshold="extreme",
        sigma_threshold=smooth
    )

    features_weighted_points = tobac.feature_detection_multithreshold(
        test_data_norm,
        threshold=[norm_threshold],  # single threshold in normalized space
        dxy=dxy,
        target="minimum",
        position_threshold="weighted_abs",
        sigma_threshold=smooth
    )


    os.makedirs(output_folder, exist_ok=True)  # create folder if it doesn't exist

    # === PLOTTING AND SAVING ===
    segments_all = []
    for i, itime in enumerate(range(0, images_no)):
        original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]
        # Smooth the frame
        smoothed_frame = ndimage.gaussian_filter(
            test_data_norm.isel(time=itime).values, sigma=smooth
        )
        temp_da = test_data_norm.isel(time=itime).copy()
        temp_da.data = smoothed_frame

        # consistent color range

        # overlay detections
        
        #------------------ segmentation ------------------  

        temp_da = test_data_norm.isel(time=[itime]).copy()
        temp_da.data = smoothed_frame[np.newaxis, ...]  # keep time dim


        field_2d = temp_da

        f = features[features["frame"] == itime]  # features in this frame
        
        if f.empty:
            print(f"No features found for frame {itime}, skipping segmentation.")
            segments_all.append((itime, None, None))
            continue

        # perform segmentation
        segment_labels, segments = tobac.segmentation_2D(
            f,
            field_2d,
            dxy=dxy,
            threshold=norm_threshold,
            target="minimum"
            
        )
        # store results
        segments_all.append((itime, segment_labels, segments))
        
   
    plot_frames = range(0, images_no)

    for i, itime in enumerate(plot_frames):
        original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]
        # Get the field for this frame
        fig, axs = plt.subplots(figsize=(6, 6))
            
        smoothed_frame = ndimage.gaussian_filter(test_data_norm.isel(time=itime).values, sigma=smooth)

        temp_da = test_data_norm.isel(time=itime).copy()
        temp_da.data = smoothed_frame

        # consistent color range across all frames
        temp_da.plot(ax=axs, vmin=0, vmax=1, cmap="viridis")
        
        f_weighted = features_weighted_points[features_weighted_points["frame"] == itime]
        f_weighted.plot.scatter(
            x="x",
            y="y",
            s=20,
            ax=axs,
            color="red",
            marker="x",
        )

        # Extract segmentation for this frame from segments_all
        # We find the entry with matching time
        entry = next((s for s in segments_all if s[0] == itime), None)
        
        if entry is not None:
            
            _, seg_labels, _ = entry
            if seg_labels is not None:
                # seg_labels may have a single-element time dimension
                seg_labels2d = seg_labels.isel(time=0)  # drop the time dim for contour
                # Only plot contour if there are actual segmented pixels
                seg_labels2d.plot.contour(levels=[0.5], ax=axs, colors="k")
            

        axs.set_title(f"Timeframe = {itime}")
        out_path = os.path.join(output_folder, f"{original_img_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # close the figure to free memory

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


def run_tobac(inpu_folder, output_folder,smooth = 8):
    
    locate_and_segment(inpu_folder, output_folder,smooth)
    print("Locating procedure completed")
    track()
    print("Tracking procedure completed")
