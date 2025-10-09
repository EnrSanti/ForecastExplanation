import os
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import tobac

# -----------------------------
# CONFIGURATION
# -----------------------------
input_dir = "to_test"        # folder with your 30 images
output_dir = "tobac_output"    # folder to save segmentation results
threshold = 0.1                # threshold for feature detection
min_size = 5                  # minimum feature size in pixels

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD FRAMES
# -----------------------------
image_files = sorted(glob(os.path.join(input_dir, "*.png")))  # keep order
frames = []
for f in image_files:
    img = imread(f, as_gray=True)
    if img.max() > 1.0:
        img = img / 255.0   # normalize to 0-1
    frames.append(img)

data = np.stack(frames, axis=0)  # shape: (time, H, W)

# -----------------------------
# CREATE DATETIME INDEX FOR TOBAC
# -----------------------------
# TOBAC expects time coordinates as datetime
times = pd.date_range("2025-01-01", periods=data.shape[0], freq="H")
da = xr.DataArray(data, dims=["time", "y", "x"], coords={"time": times})

print(da.min().values, da.max().values)

# -----------------------------
# SPACINGS (manual, since no coordinates)
# -----------------------------
dxy = 1.0  # pixel spacing
dt = 1.0   # time spacing (hour between frames, consistent with times above)

# -----------------------------
# FEATURE DETECTION
# -----------------------------
features = tobac.feature_detection_multithreshold(da, dxy, threshold)

# -----------------------------
# SEGMENTATION
# -----------------------------
segment_labels, segments_df = tobac.segmentation_2D(features, da, dxy, threshold=threshold)

# -----------------------------
# SAVE SEGMENTED MASKS
# -----------------------------
for t in range(da.shape[0]):
    mask = segment_labels.isel(time=t)
    out_path = os.path.join(output_dir, f"frame_{t:03d}.png")
    imsave(out_path, mask.astype(np.uint16))

print(f"[INFO] Segmentation complete. Labeled masks saved in {output_dir}")

# -----------------------------
# OPTIONAL: visualize first few frames
# -----------------------------
fig, axs = plt.subplots(ncols=1, nrows=min(3, da.shape[0]), figsize=(12, 16), sharey=True)
plt.subplots_adjust(hspace=0.5)

for i, itime in enumerate([0, 10, 20]):  # adjust frames to visualize
    da.isel(time=itime).plot(ax=axs[i])
    segment_labels.isel(time=itime).plot.contour(levels=[0.5], ax=axs[i], colors="k")
    axs[i].set_title(f"timeframe = {itime}")
