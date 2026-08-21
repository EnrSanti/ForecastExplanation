import gc
import logging
import os
import re

import cv2
import imageio
import imageio as images
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import seaborn as sns
import tobac
import tobac.testing
import xarray as xr

from image_processing.split_merge import plot_feature_borders, find_extended_overlap_blobs_inferred, get_splits_merges

logging.getLogger("trackpy").setLevel(logging.WARNING)

DEBUG = False  # True would print the circles and search radius


def overlay_image(path_borders, axs, temp_da):
    img = plt.imread(path_borders)
    axs.imshow(img, extent=(0, temp_da.sizes["x"], temp_da.sizes["y"], 0), alpha=0.6)


def get_blob_positions(trajectories, itime):
    cells_prev_step = []
    itime_prev = itime - 1
    prev_frames_data = trajectories[trajectories["frame"].between(itime_prev, itime_prev)]
    cells_prev_step = list(prev_frames_data["cell"].unique())

    cells_prev_step = list(cells_prev_step)
    cell_ids_in_frame = trajectories[trajectories["frame"] == itime]["cell"].unique()
    # Create necessary mappings and lists

    all_moved_cells = ""
    for cell_id in cells_prev_step:
        if (cell_id in cell_ids_in_frame):
            # previous positions
            prev_x = \
                trajectories[((trajectories["frame"] == itime_prev) & (trajectories["cell"] == cell_id))]["x"].iloc[0]
            prev_y = \
                trajectories[((trajectories["frame"] == itime_prev) & (trajectories["cell"] == cell_id))]["y"].iloc[0]
            # current positions
            curr_x = trajectories[((trajectories["frame"] == itime) & (trajectories["cell"] == cell_id))]["x"].iloc[0]
            curr_y = trajectories[((trajectories["frame"] == itime) & (trajectories["cell"] == cell_id))]["y"].iloc[0]
            all_moved_cells += f"Frame {itime}, cell {cell_id} moved from (x: {prev_x}, y:{prev_y} ) to (x: {curr_x}, y:{curr_y} )\n"

    return all_moved_cells


# target minimum -> upper
# target maximum -> lower
def locate_track_merge(input_folder, output_folder, border_path, n_min_threshold, lat_min, lat_max, lon_min, lon_max,
                       threshold, target, type_, save_split_merges=True, smooth=8):
    """
    Runs the locate and tracking of the objects
    
    Parameters
    ----------
    inpu_folder: folder path containing the input images (equal size & regional area)
    output_folder: folder path to save the output images (if not existing it will be created)
    lat_min: the minimum latitude of the area in the images
    lat_max: the maximum latitude of the area in the images
    lon_min: the minimum longitude of the area in the images
    lon_max: the maximum longitude of the area in the images
    n_min_threshold: minimum number of pixels for object detection (default 0)
    smooth: smoothing factor for gaussian filter (default 8)
    """

    # create folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load images from input folder
    image_files = ([os.path.join(input_folder, f) for f in os.listdir(input_folder)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    images_no = len(image_files)
    image_files = sorted(image_files, key=extract_keys)
    frames = [imageio.v2.imread(f) for f in image_files]

    # Extract datetimes from filenames (will be put in the dataframe)
    datetimes = []
    for f in image_files:
        basename = os.path.basename(f)
        # eg "cloud_123_20251008_1200.png", split by underscore and take the last two parts
        parts = basename.split("_")
        date_str = parts[-2]  # YYYYMMDD
        hour_str = parts[-1].split(".")[0]  # HHHH
        dt_str = date_str + hour_str  # "YYYYMMDDHHHH"

        # convert to pandas datetime
        dt = pd.to_datetime(dt_str, format="%Y%m%d%H%M")  # assuming HHHH is HHMM
        datetimes.append(dt)

    # convert to array
    datetimes = pd.to_datetime(datetimes)

    # convert frames to grayscale
    frames_gray = []
    if type_ in ["temp"]:
        frames_gray = [1 - np.mean(f[:, :, :3], axis=2) if f.ndim == 3 else f for f in frames]
    else:
        frames_gray = [np.mean(f[:, :, :3], axis=2) if f.ndim == 3 else f for f in frames]

    # stack into 3D array (time, y, x)
    data = np.stack(frames_gray)
    _, n_y, n_x = data.shape

    # spatial coordinates (example: 1 pixel = 1000 m)
    x = np.arange(n_x)
    y = np.arange(n_y)

    # create xarray.DataArray with the time info
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

    # works on fvg coordinates (for now), i set the lat/lon based on the image size
    lat = np.linspace(lat_min, lat_max, n_y)
    lon = np.linspace(lon_min, lon_max, n_x)
    latitude = np.tile(lat[:, np.newaxis], (1, n_x))
    longitude = np.tile(lon[np.newaxis, :], (n_y, 1))
    test_data = test_data.assign_coords(latitude=(("y", "x"), latitude),
                                        longitude=(("y", "x"), longitude))

    # run tobac to get the spacings
    dxy, dt = tobac.get_spacings(test_data, grid_spacing=(1, 1))

    # normalize all data in the different plots so we can use a single scale/legend and threshold
    vmin = float(test_data.min())
    vmax = float(test_data.max())
    test_data_norm = (test_data - vmin) / (vmax - vmin)

    # === FEATURE DETECTION ===
    # Locate twice just to get the segmentation right (i.e. with "extreme" i know the center will be inside the object)
    features = tobac.feature_detection_multithreshold(
        test_data_norm,
        threshold=[threshold],  # single threshold in normalized space
        dxy=3000,  # 1 px 3km
        target=target,
        position_threshold="extreme",
        sigma_threshold=smooth,
        n_min_threshold=n_min_threshold,
        min_distance=1000  # at least 500m between 2 objects
    )
    # this will be used for getting the center of the objects, the one above for segmentation
    features_weighted_points = tobac.feature_detection_multithreshold(
        test_data_norm,
        threshold=[threshold],
        dxy=3000,
        target=target,
        position_threshold="weighted_abs",
        sigma_threshold=smooth,
        n_min_threshold=n_min_threshold,
        min_distance=1000  # at least 1000m between 2 objects
    )

    dt = 3600
    dxy = 2500
    v_max = 70
    gap_features_frames = 1  # for how many frames a feature can disappear and still be linked (2 full frames in this case, it reappers in the 3)
    radius = v_max * dt / dxy

    # ======== FEATURE TRACKING ========
    # using predict, i may be a little bit out of the "search raius" but ok

    try:
        trajectories = tobac.linking_trackpy(features_weighted_points, test_data, dt=dt, dxy=dxy, v_max=v_max,
                                             memory=gap_features_frames, method_linking="predict")
    except Exception as e:
        print("No trajectories found: ")
        trajectories = None

    # ======== SEGMENTING ========

    segments_all = []
    all_segment_labels = []
    new_born_at_curr = {}
    disappeared_at_curr = {}
    all_frames_for_cell = {}

    plot_frames = range(0, images_no)

    # getting how to color the images
    cmap = "viridis"
    if type_ == "cloud":
        cmap = "viridis"
    elif type_ == "humidity":
        cmap = "YlGnBu"
    elif type_ == "temp":
        cmap = "OrRd"

    # particular case: no trajectories found, just plot the smoothed images
    if (trajectories is None):
        for i, itime in enumerate(plot_frames):
            original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]
            fig_width_in = n_x / 100
            fig_height_in = n_y / 100
            fig, axs = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=100)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # print("Figure size in pixels:", int(fig_width_px), int(fig_height_px))
            smoothed_frame = ndimage.gaussian_filter(test_data_norm.isel(time=itime).values, sigma=smooth)

            temp_da = test_data_norm.isel(time=itime).copy()
            temp_da.data = smoothed_frame

            # consistent color range across all frames
            axs.imshow(temp_da.values, origin="upper", cmap=cmap)  # pixels are axes
            xlim = (0, temp_da.sizes['x'])
            ylim = (0, temp_da.sizes['y'])
            axs.set_title("")
            axs.set_xticks([])  # remove x-axis ticks
            axs.set_yticks([])  # remove y-axis ticks
            axs.set_xticklabels([])  # remove x-axis labels
            axs.set_yticklabels([])  # remove y-axis labels
            axs.axis('off')
            out_path = os.path.join(output_folder, f"{original_img_name}.png")
            axs.set_xlim(0, temp_da.sizes["x"])
            axs.set_ylim(temp_da.sizes["y"], 0)  # since origin="upper"

            # add the fvg borders
            overlay_image(border_path, axs, temp_da)

            plt.savefig(out_path, dpi=100, bbox_inches=None, pad_inches=0)
            plt.close(fig)
        return

    # for all images i smooth the frame and collect the segments
    for i, itime in enumerate(range(0, images_no)):
        original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]
        # Smooth the frame
        smoothed_frame = ndimage.gaussian_filter(
            test_data_norm.isel(time=itime).values, sigma=smooth
        )

        temp_da = test_data_norm.isel(time=[itime]).copy()
        temp_da.data = smoothed_frame[np.newaxis, ...]  # keep time dim

        # features in this frame
        f = features[features["frame"] == itime]

        if f.empty:
            # print(f"No features found for frame {itime}, skipping segmentation.")
            segments_all.append((itime, None, None))
            all_segment_labels.append(None)
            continue

        # perform segmentation
        segment_labels, segments = tobac.segmentation_2D(
            f,
            temp_da,
            dxy=dxy,
            threshold=threshold,
            target=target
        )

        # store results
        segments_all.append((itime, segment_labels, segments))
        all_segment_labels.append(segment_labels)

    # ===== PLOTTING =====

    # keep track of cells in previous frames for "gap_features_frames" frames
    cells_frames_before = []
    # what cells are in the current frame
    cell_ids = set()
    for i, itime in enumerate(plot_frames):

        # remove the oldest frame and add the current one
        # cells_frames_before.pop(0)

        # get the cells in this frame
        cell_ids = set(trajectories[(trajectories["frame"] == itime)]["cell"].dropna().unique())

        # all_cells_in_gap collects all the cells in the previous gap_features_frames frames (a cell may have disappeared and reappeared)
        all_cells_in_gap = set()
        # map containing a list, for each cell in the current frame: all the frames it appeared in the previous gap_features_frames frames
        all_frames_for_cell = {}

        for j in range(gap_features_frames + 1):
            # set sum
            if (i - j - 1 >= 0):
                all_cells_in_gap = all_cells_in_gap | cells_frames_before[i - j - 1]
                print("frame ", i, " cell ids: ", cell_ids, " cells in gap (frame ", i - j - 1, " )",
                      cells_frames_before[i - j - 1])

                for el in cells_frames_before[i - j - 1]:
                    if el not in all_frames_for_cell:
                        all_frames_for_cell[el] = []  # create a list for new cells
                    # add the frame number where the cell appeared
                    all_frames_for_cell[el].append(i - j - 1)

        persisted = cell_ids & all_cells_in_gap  # intersection -> clouds present now and previously
        new_cells = cell_ids - all_cells_in_gap  # new this frame (may have reappeared after too long (> gap_features_frames))
        disappeared = all_cells_in_gap - cell_ids  # disappeared clouds in this frame

        new_born_at_curr[itime] = new_cells
        disappeared_at_curr[itime] = disappeared

        original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]

        # Get the field for this frame

        fig_width_in = n_x / 100
        fig_height_in = n_y / 100
        fig, axs = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # print("Figure size in pixels:", int(fig_width_px), int(fig_height_px))
        smoothed_frame = ndimage.gaussian_filter(test_data_norm.isel(time=itime).values, sigma=smooth)

        temp_da = test_data_norm.isel(time=itime).copy()
        temp_da.data = smoothed_frame

        # consistent color range across all frames
        axs.imshow(temp_da.values, origin="upper", cmap=cmap)  # pixels are axes
        xlim = (0, temp_da.sizes['x'])
        ylim = (0, temp_da.sizes['y'])

        # forall cells in this frame plot the trajectory, center, segmentation and radius
        for cell_id in cell_ids:
            # get if the cell id is in the current frame
            track = trajectories[trajectories["cell"] == cell_id]
            f_weighted = track[(track["frame"] == itime)]

            printing_symbol = ''
            color = ''

            if (cell_id in new_cells):
                printing_symbol = '^'
                color = 'white'
            else:
                printing_symbol = 'x'
                color = 'red'

            # print trajectory and center (if new or persisted the symbol changes)
            print_clouds_center_line(printing_symbol, color, f_weighted, itime, track, axs, cell_id, persisted,
                                     all_frames_for_cell)

            if (len(f_weighted["x"]) == 0):
                continue

            # print cell id numbers on the plot for clarity
            print_cloud_labels(f_weighted, cell_id, xlim, ylim, axs)
            # print the radius
            if (DEBUG):
                add_circle_slice_filled(axs, f_weighted, radius=radius, xlim=xlim, ylim=ylim, color='red', alpha=0.05)

        # Extract segmentation for this frame from segments_all and print it
        entry = next((s for s in segments_all if s[0] == itime), None)

        if entry is not None:
            _, seg_labels, segment_labels_ = entry
            if seg_labels is not None:
                # seg_labels may have a single-element time dimension
                seg_labels2d = seg_labels.isel(time=0)  # drop the time dim for contour
                # Only plot contour if there are actual segmented pixels
                seg_labels2d.plot.contour(levels=[0.5], ax=axs, colors="k")

        # finalize the figure

        axs.set_title("")
        axs.set_xticks([])  # remove x-axis ticks
        axs.set_yticks([])  # remove y-axis ticks
        axs.set_xticklabels([])  # remove x-axis labels
        axs.set_yticklabels([])  # remove y-axis labels
        axs.axis('off')
        out_path = os.path.join(output_folder, f"{original_img_name}.png")
        axs.set_xlim(0, temp_da.sizes["x"])
        axs.set_ylim(temp_da.sizes["y"], 0)  # since origin="upper"

        overlay_image(border_path, axs, temp_da)

        plt.savefig(out_path, dpi=100, bbox_inches=None, pad_inches=0)
        plt.close(fig)
        cells_frames_before.append(cell_ids)
    print("all frames for: ", all_frames_for_cell)

    # ======= SPLITTING AND MERGING ======== #for tthe clouds
    if (save_split_merges):
        blob_positions = ""
        all_splits_merges = ""
        for i, itime in enumerate(range(1, images_no)):
            if (all_segment_labels[itime - 1] is None or all_segment_labels[itime] is None):
                continue

            # unused if not for debugging
            plot_feature_borders(
                segment_labels=all_segment_labels[i].isel(time=0).values,
                ax=axs,
                border_thickness_px=8,  # Example: 3 pixels thick border
                border_color="red"
            )

            extended_overlap_map = find_extended_overlap_blobs_inferred(
                segment_labels=all_segment_labels[i].isel(time=0).values,
                trajectories=trajectories,
                border_thickness_px=8  # Use the same thickness as your plot border
            )

            splits, merges = get_splits_merges(extended_overlap_map, trajectories, itime, images_no,
                                               gap_features_frames, all_segment_labels[itime],
                                               all_segment_labels[itime - 1], new_born_at_curr[itime],
                                               disappeared_at_curr[itime])

            blob_positions += get_blob_positions(trajectories, itime)

            if splits != "" or merges != "":
                all_splits_merges += splits + merges
                all_splits_merges += "-------------------\n"

            # print("Consideering frame ---------------------------- ", itime+1)
        gc.collect()
        # save the merge and splits found:
        with open(output_folder + f"/split_merge.txt", "w") as f:
            f.write(str(all_splits_merges))
        with open(output_folder + f"/movements.txt", "w") as f:
            f.write(str(blob_positions))

    # save the trajectories
    trajectories.to_csv(output_folder + f"/trajectories.csv", index=False)

    # save the segments_all:
    np.savez_compressed(output_folder + f"/segment_labels_all.npz", *all_segment_labels)


def print_clouds_center_line(printing_symbol, color, f_weighted, itime, track, axs, cell_id, persisted_cells,
                             all_frames_for_cell):
    """
    Prints on the plot (axs) the trace and center of the cloud specified by cell_id at frame itime. 
    
    Parameters
    ----------
    printing_symbol: symbol marking the center of the cloud (it's different if the cloud just appeared or was already present in previous frames)
    color: different color if the cloud just appeared or was already present in previous frames
    f_weighted: cloud data for this specific frame
    itime: current frame index
    track: full trajectory data for the single cloud (in all time steps)
    axs: matplotlib axes where to plot
    cell_id: the id of the cloud to plot
    persisted_cells: set of cell ids that were already present in previous frames
    all_frames_for_cell: map of cell id to list of frames where the cell appeared in the previous gap_features_frames frames

    """
    cell_in_this_frame = not (track[track["frame"] == itime].empty)

    # if the cloud persisted from previous frames, print the last segment of the trajectory
    if (cell_id in persisted_cells and cell_in_this_frame):
        last_frame = (all_frames_for_cell[cell_id])[-1]
        line = track[(track["frame"] == last_frame) | (track["frame"] == itime)]  # last two frames

        # plot main trajectory (last step)
        axs.plot(
            line["x"],
            line["y"],
            color="blue",
            linewidth=1.5,
            alpha=0.5,
        )

    # plot trajectory with gradient (fading older traces)
    try:
        frames = all_frames_for_cell[int(cell_id)]
        # for all pair of frames
        for t0, t1 in zip(frames[:-1], frames[1:]):
            line = track[(track["frame"] == t0) | (track["frame"] == t1)]
            alpha = 0.1 + 0.3 * (t0 - track.iloc[0].frame) / (itime - track.iloc[0].frame)

            axs.plot(
                line["x"],
                line["y"],
                color="blue",
                linewidth=1.5,
                alpha=alpha,
            )

    except KeyError:
        pass
        # print("no frames for cell ", cell_id)

    # The cloud exists at this frame, mark the center
    f_weighted.plot.scatter(
        x="x",
        y="y",
        s=40,
        ax=axs,
        color=color,
        marker=printing_symbol,
    )


def print_cloud_labels(f_weighted, cell_id, xlim, ylim, axs):
    """
    Prints on the plot (axs) the label (cell_id) associated to each blob, just for clarity. 
    
    Parameters
    ----------
    f_weighted: cloud data for this specific frame
    cell_id: the id of the cloud label to plot
    xlim: x limits of the plot (to keep all output images the same size and avoid cutting labels)
    ylim: y limits of the plot (to keep all output images the same size and avoid cutting labels)
    axs: matplotlib axes where to plot
    
    """
    # get the position and adjust if too close to the edges
    x_pos = f_weighted["x"].values[0]
    y_pos = f_weighted["y"].values[0]

    if x_pos < xlim[0] + 30:
        x_pos = x_pos + 20
    if x_pos > xlim[1] - 30:
        x_pos = x_pos - 20

    if y_pos < ylim[0] + 30:
        y_pos = y_pos + 20
    if y_pos > ylim[1] - 30:
        y_pos = y_pos - 20

    axs.text(
        x_pos - 3,  # offset a bit to the right
        y_pos - 3,  # offset upward slightly
        f"{int(cell_id)}",  # text = cloud id
        color="white",
        fontsize=8,
        weight="bold",
        bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=1)
    )


def add_circle_slice_filled(ax, f_weighted, radius, xlim, ylim, color="red", alpha=0.5, **kwargs):
    """
    Prints on the plot (axs) the circle associated to the linking of each blob (i.e. where in the next frame the cloud will be looked for), just for clarity. 
    
    Parameters
    ----------
    ax: matplotlib axes where to plot
    cx: x center of the circle
    cy: y center of the circle
    radius: radius of the circle
    xlim: x limits of the plot (to keep all output images the same size and avoid cutting labels)
    ylim: y limits of the plot (to keep all output images the same size and avoid cutting labels)
    color: color of the circle
    alpha: transparency of the circle
    kwards: additional arguments for matplotlib.patches.Polygon

    """

    # get the position (center for the circle)
    cx = f_weighted["x"].iloc[0]
    cy = f_weighted["y"].iloc[0]

    # sample circle points
    theta = np.linspace(0, 2 * np.pi, 300)
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    # clip points to the plot
    x_clipped = np.clip(x, xlim[0], xlim[1])
    y_clipped = np.clip(y, ylim[0], ylim[1])

    # polygon from clipped points
    polygon_points = np.column_stack([x_clipped, y_clipped])

    # add corners if any clipped point is at box edge
    corners = []
    if np.any(x < xlim[0]) and np.any(y < ylim[0]):
        corners.append([xlim[0], ylim[0]])
    if np.any(x > xlim[1]) and np.any(y < ylim[0]):
        corners.append([xlim[1], ylim[0]])
    if np.any(x < xlim[0]) and np.any(y > ylim[1]):
        corners.append([xlim[0], ylim[1]])
    if np.any(x > xlim[1]) and np.any(y > ylim[1]):
        corners.append([xlim[1], ylim[1]])

    # just plotting
    if corners:
        polygon_points = np.vstack([polygon_points, corners])

    polygon = patches.Polygon(polygon_points, closed=True,
                              facecolor=color, alpha=alpha, **kwargs)

    ax.add_patch(polygon)

    polygon_border = patches.Polygon(polygon_points, closed=True,
                                     facecolor="none", alpha=0.3, edgecolor="red", linestyle="--", linewidth=1,
                                     **kwargs)
    ax.add_patch(polygon_border)


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


def run_tobac_merge_split(inpu_folder, output_folder, border_path, lat_min, lat_max, lon_min, lon_max, threshold,
                          target, type_, n_min_threshold=0, smooth=8):
    """
    The main function called from outside (main).
    Runs the locate and tracking of the objects, moreover it does the splitting and merging detection.
    
    Parameters
    ----------
    inpu_folder: folder path containing the input images (equal size & regional area)
    output_folder: folder path to save the output images (if not existing it will be created)
    lat_min: the minimum latitude of the area in the images
    lat_max: the maximum latitude of the area in the images
    lon_min: the minimum longitude of the area in the images
    lon_max: the maximum longitude of the area in the images
    n_min_threshold: minimum number of pixels for object detection (default 0)
    smooth: smoothing factor for gaussian filter (default 8)

    """
    locate_track_merge(inpu_folder, output_folder, border_path, n_min_threshold, lat_min, lat_max, lon_min, lon_max,
                       threshold, target, type_, True, smooth)
    print("Locating & tracking procedure completed")


def run_tobac_fronts(inpu_folder, output_folder, border_path, lat_min, lat_max, lon_min, lon_max, threshold, target,
                     type_, n_min_threshold=0, smooth=8):
    """
    The main function called from outside (main).
    Runs the locate and tracking of the objects, it doens't do the splitting and merging detection, but tracks fronts.
    
    Parameters
    ----------
    inpu_folder: folder path containing the input images (equal size & regional area)
    output_folder: folder path to save the output images (if not existing it will be created)
    lat_min: the minimum latitude of the area in the images
    lat_max: the maximum latitude of the area in the images
    lon_min: the minimum longitude of the area in the images
    lon_max: the maximum longitude of the area in the images
    n_min_threshold: minimum number of pixels for object detection (default 0)
    smooth: smoothing factor for gaussian filter (default 8)

    """
    locate_track_merge(inpu_folder, output_folder, border_path, n_min_threshold, lat_min, lat_max, lon_min, lon_max,
                       threshold, target, type_, False, smooth)
    print("Locating & tracking (fronts) procedure completed")
