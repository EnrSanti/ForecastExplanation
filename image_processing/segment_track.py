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
import matplotlib.patches as patches
import imageio as images
sns.set_context("talk")

def locate_track(input_folder, output_folder,n_min_threshold,lat_min,lat_max,lon_min,lon_max,smooth = 8):
   
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
    _, n_y, n_x = data.shape

    # Spatial coordinates (example: 1 pixel = 1000 m)
    dx = dy = 3000  
    x = np.arange(n_x)
    y = np.arange(n_y)

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
    #works on fvg coordinates
    lat = np.linspace(lat_min, lat_max, n_y)
    lon = np.linspace(lon_min, lon_max, n_x)
    latitude = np.tile(lat[:, np.newaxis], (1, n_x))
    longitude = np.tile(lon[np.newaxis, :], (n_y, 1))
    test_data = test_data.assign_coords(latitude=(("y","x"), latitude),
                                        longitude=(("y","x"), longitude))
    

    dxy, dt = tobac.get_spacings(test_data,grid_spacing=(1, 1))

    print("dxy:", dxy)
    print("dt:", dt)

    # === GLOBAL NORMALIZATION ===
    vmin = float(test_data.min())
    vmax = float(test_data.max())
    test_data_norm = (test_data - vmin) / (vmax - vmin)


    # Convert original threshold (155) to normalized [0, 1] scale
    norm_threshold = 0.75


    # === FEATURE DETECTION ===
    #I locate twice just to get the segmentation right (i.e. with "extreme" i know the center will be inside the object)
    features = tobac.feature_detection_multithreshold(
        test_data_norm,
        threshold=[norm_threshold],  # single threshold in normalized space
        dxy=3000,  # 1 px 3km
        target="minimum",
        position_threshold="extreme",
        sigma_threshold=smooth,
        n_min_threshold=n_min_threshold,
        min_distance=500 # i guess 500m
    )

    features_weighted_points = tobac.feature_detection_multithreshold(
        test_data_norm,
        threshold=[norm_threshold],  # single threshold in normalized space
        dxy=3000, # 1 px 3km
        target="minimum",
        position_threshold="weighted_abs",
        sigma_threshold=smooth,
        n_min_threshold=n_min_threshold,
        min_distance=500 #i guess 500m
    )
    dt=3600
    dxy=3000
    v_max=100
    trajectories = tobac.linking_trackpy(features_weighted_points, test_data, dt=dt, dxy=dxy, v_max=v_max)

    os.makedirs(output_folder, exist_ok=True)  # create folder if it doesn't exist

    radius=v_max*dt/dxy



    #======== SEGMENTING ========
    segments_all = []
    for i, itime in enumerate(range(0, images_no)):
        original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]
        # Smooth the frame
        smoothed_frame = ndimage.gaussian_filter(
            test_data_norm.isel(time=itime).values, sigma=smooth
        )
        temp_da = test_data_norm.isel(time=itime).copy()
        temp_da.data = smoothed_frame

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

    #======== plotting ====
    cell_ids = features["idx"].dropna().unique()

    for i, itime in enumerate(plot_frames):

        original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]
        # Get the field for this frame
        fig, axs = plt.subplots(figsize=(6, 6))
            
        smoothed_frame = ndimage.gaussian_filter(test_data_norm.isel(time=itime).values, sigma=smooth)

        temp_da = test_data_norm.isel(time=itime).copy()
        temp_da.data = smoothed_frame

        # consistent color range across all frames
        axs.imshow(temp_da.values, origin="upper", cmap="viridis")  # pixels are axes
        xlim = (0, temp_da.sizes['x'])
        ylim = (0, temp_da.sizes['y'])

        for cell_id in cell_ids:
            track = trajectories[trajectories["cell"] == cell_id]
            f_weighted = features_weighted_points[(features_weighted_points["frame"] == itime) & (features_weighted_points["idx"] == cell_id)]
            print_clouds_center(f_weighted,features_weighted_points, itime, track, axs,cell_id)      
            
            #print cell id numbers
            if(len(f_weighted["x"])==0):
                continue
            print_cloud_labels(f_weighted, cell_id,xlim, ylim, axs)
            add_circle_slice_filled(axs, cx=f_weighted["x"].iloc[0], cy=f_weighted["y"].iloc[0], radius=radius, xlim=xlim, ylim=ylim,color='red', alpha=0.05)

        # Extract segmentation for this frame from segments_all
            
        entry = next((s for s in segments_all if s[0] == itime), None)
        
        if entry is not None:
            
            _, seg_labels, _ = entry
            if seg_labels is not None:
                # seg_labels may have a single-element time dimension
                seg_labels2d = seg_labels.isel(time=0)  # drop the time dim for contour
                # Only plot contour if there are actual segmented pixels
                seg_labels2d.plot.contour(levels=[0.5], ax=axs, colors="k")
                



        axs.set_title("")
        #axs.set_title(f"Timeframe = {itime}")
        axs.set_xticks([])       # remove x-axis ticks
        axs.set_yticks([])       # remove y-axis ticks
        axs.set_xticklabels([])  # remove x-axis labels
        axs.set_yticklabels([])  # remove y-axis labels
        axs.axis('off')    
        out_path = os.path.join(output_folder, f"{original_img_name}.png")
        axs.set_xlim(0, temp_da.sizes["x"])
        axs.set_ylim(temp_da.sizes["y"], 0)  # since origin="upper"
        plt.savefig(out_path, dpi=150, bbox_inches="tight",pad_inches=0)
        plt.close(fig)  # close the figure to free memory
    

def print_clouds_center(f_weighted, features_weighted_points, itime, track, axs, i,
                        dt=3600, dxy=3000, v_max=100):
    """
    Plot a single cloud and its trajectory line, drawing the actual search radius circle.
    """
    # compute linking search radius in pixels
    search_radius = v_max * dt / dxy

    line = track[(track["frame"] == itime - 1) | (track["frame"] == itime)]  # last two frames

    # stop if the blob doesn't exist yet
    if track.iloc[-1].frame < itime:
        return

    # plot main trajectory
    axs.plot(
        line["x"],
        line["y"],
        color="red",
        linewidth=1.5,
        alpha=0.5,
    )

    # plot trajectory gradient (fading older)
    for jtime in range(track.iloc[0].frame, itime - 1):
        line = track[(track["frame"] == jtime) | (track["frame"] == jtime + 1)]
        alpha = 0.1 + 0.3 * (jtime - track.iloc[0].frame) / (itime - track.iloc[0].frame)
        axs.plot(
            line["x"],
            line["y"],
            color="red",
            linewidth=1.5,
            alpha=alpha,
        )

    # if the cloud exists at this frame, get its coordinates
   

    # draw scatter for the cloud and print info
    if len(line) <= 1:  # new cloud
        f_weighted.plot.scatter(
            x="x",
            y="y",
            s=40,
            ax=axs,
            color="white",
            marker="^",
        )
        
    else:
        f_weighted.plot.scatter(
            x="x",
            y="y",
            s=40,
            ax=axs,
            color="red",
            marker="x",
        )
        print(f"Tracked cloud (len={len(line)}) at frame {itime} (id={i}) | search radius = {search_radius:.2f} px")


def print_cloud_labels(f_weighted, cell_id,xlim, ylim, axs):
    
    x_pos = f_weighted["x"].values[0]
    y_pos = f_weighted["y"].values[0]

    if x_pos < xlim[0]+30:
        x_pos = x_pos+20
    if x_pos > xlim[1]-30:
        x_pos = x_pos-20
    
    if y_pos < ylim[0]+30:
        y_pos = y_pos+20
    if y_pos > ylim[1]-30:
        y_pos = y_pos-20

    axs.text(
        x_pos -3,  # offset a bit to the right
        y_pos -3,  # offset upward slightly
        f"{int(cell_id)}",  # text = cloud id
        color="white",
        fontsize=8,
        weight="bold",
        bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=1)
    )
def add_circle_slice_filled(ax, cx, cy, radius, xlim, ylim, color="red", alpha=0.5, **kwargs):
    """
    Draw a filled 'slice' of a circle that stays within xlim/ylim.
    Original radius is preserved, parts outside the limits are clipped.
    Corners are filled with triangles to handle circle at edges/corners.
    """
    # Sample circle points
    theta = np.linspace(0, 2*np.pi, 300)
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    # Clip points to the box
    x_clipped = np.clip(x, xlim[0], xlim[1])
    y_clipped = np.clip(y, ylim[0], ylim[1])

    # Create polygon from clipped points
    polygon_points = np.column_stack([x_clipped, y_clipped])

    # Add corners if any clipped point is at box edge
    corners = []
    if np.any(x < xlim[0]) and np.any(y < ylim[0]):
        corners.append([xlim[0], ylim[0]])
    if np.any(x > xlim[1]) and np.any(y < ylim[0]):
        corners.append([xlim[1], ylim[0]])
    if np.any(x < xlim[0]) and np.any(y > ylim[1]):
        corners.append([xlim[0], ylim[1]])
    if np.any(x > xlim[1]) and np.any(y > ylim[1]):
        corners.append([xlim[1], ylim[1]])

    if corners:
        polygon_points = np.vstack([polygon_points, corners])

    polygon = patches.Polygon(polygon_points, closed=True,
                              facecolor=color, alpha=alpha,**kwargs)
    
    ax.add_patch(polygon)

    polygon_border = patches.Polygon(polygon_points, closed=True,
                              facecolor="none", alpha=0.3,edgecolor="red",linestyle="--",linewidth=1,**kwargs)
    ax.add_patch(polygon_border)


def extract_keys(filename):
    # match pattern like cloud_200_YYYYMMDD_HHMM.png
    m = re.search(r'_(\d{8})_(\d+)\.png$', filename)
    if m:
        date = int(m.group(1))
        num = int(m.group(2))
        return (date, num)
    else:
        return (0, 0)


def run_tobac(inpu_folder, output_folder,lat_min,lat_max,lon_min,lon_max,n_min_threshold=0,smooth = 8):
    
    locate_track(inpu_folder, output_folder,n_min_threshold,lat_min,lat_max,lon_min,lon_max,smooth)
    print("Locating & tracking procedure completed")
