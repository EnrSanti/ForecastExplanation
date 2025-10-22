import tobac
import imageio
import os
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


def locate_track(input_folder, output_folder,n_min_threshold,lat_min,lat_max,lon_min,lon_max,smooth = 8):
    
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

    #Load images from input folder
    image_files = ([os.path.join(input_folder, f) for f in os.listdir(input_folder)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    images_no=len(image_files)
    image_files = sorted(image_files, key=extract_keys)
    frames = [imageio.v2.imread(f) for f in image_files]

    #Extract datetimes from filenames (will be put in the dataframe)
    datetimes = []
    for f in image_files:
        basename = os.path.basename(f)
        #eg "cloud_123_20251008_1200.png", split by underscore and take the last two parts
        parts = basename.split("_")
        date_str = parts[-2]      #YYYYMMDD
        hour_str = parts[-1].split(".")[0]  #HHHH
        dt_str = date_str + hour_str        #"YYYYMMDDHHHH"
        
        #convert to pandas datetime
        dt = pd.to_datetime(dt_str, format="%Y%m%d%H%M")  # assuming HHHH is HHMM
        datetimes.append(dt)

    #convert to array
    datetimes = pd.to_datetime(datetimes)

    #convert frames to grayscale
    frames_gray = [np.mean(f[:, :, :3], axis=2) if f.ndim==3 else f for f in frames]

    #stack into 3D array (time, y, x)
    data = np.stack(frames_gray)
    _, n_y, n_x = data.shape

    #spatial coordinates (example: 1 pixel = 1000 m)
    dx = dy = 3000  
    x = np.arange(n_x)
    y = np.arange(n_y)

    #create xarray.DataArray with the time info
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

    #works on fvg coordinates (for now), i set the lat/lon based on the image size
    lat = np.linspace(lat_min, lat_max, n_y)
    lon = np.linspace(lon_min, lon_max, n_x)
    latitude = np.tile(lat[:, np.newaxis], (1, n_x))
    longitude = np.tile(lon[np.newaxis, :], (n_y, 1))
    test_data = test_data.assign_coords(latitude=(("y","x"), latitude),
                                        longitude=(("y","x"), longitude))
    

    #run tobac to get the spacings
    dxy, dt = tobac.get_spacings(test_data,grid_spacing=(1, 1))

    #normalize all data in the different plots so we can use a single scale/legend and threshold
    vmin = float(test_data.min())
    vmax = float(test_data.max())
    test_data_norm = (test_data - vmin) / (vmax - vmin)


    #original threshold was 155 to normalized [0, 1]:
    norm_threshold = 0.75


    # === FEATURE DETECTION ===
    #Locate twice just to get the segmentation right (i.e. with "extreme" i know the center will be inside the object)
    features = tobac.feature_detection_multithreshold(
        test_data_norm,
        threshold=[norm_threshold],  #single threshold in normalized space
        dxy=3000,  #1 px 3km
        target="minimum",
        position_threshold="extreme",
        sigma_threshold=smooth,
        n_min_threshold=n_min_threshold,
        min_distance=500 #at least 500m between 2 objects
    )
    #this will be used for getting the center of the objects, the one above for segmentation
    features_weighted_points = tobac.feature_detection_multithreshold(
        test_data_norm,
        threshold=[norm_threshold], 
        dxy=3000,
        target="minimum",
        position_threshold="weighted_abs",
        sigma_threshold=smooth,
        n_min_threshold=n_min_threshold,
        min_distance=500 #at least 500m between 2 objects
    )

    
    dt=3600
    dxy=3000
    v_max=100
    gap_features_frames=3 #for how many frames a feature can disappear and still be linked (2 full frames in this case, it reappers in the 3)
    radius=v_max*dt/dxy

    #======== FEATURE TRACKING ========
    #using predict, i may be a little bit out of the "search raius" but ok
    trajectories = tobac.linking_trackpy(features_weighted_points, test_data, dt=dt, dxy=dxy, v_max=v_max,method_linking="predict", memory=gap_features_frames)

    #create folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)  

    #======== MERGING SPLITTING ========
    
    split_and_merge(trajectories,dxy,os.path.join(output_folder, "merge_split_info.txt"))  

   
    #======== SEGMENTING ========

    segments_all = []
    #for all images i smooth the frame and collect the segments
    for i, itime in enumerate(range(0, images_no)):
        original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]
        #Smooth the frame
        smoothed_frame = ndimage.gaussian_filter(
            test_data_norm.isel(time=itime).values, sigma=smooth
        )
        temp_da = test_data_norm.isel(time=itime).copy()
        temp_da.data = smoothed_frame

        temp_da = test_data_norm.isel(time=[itime]).copy()
        temp_da.data = smoothed_frame[np.newaxis, ...]  #keep time dim


        field_2d = temp_da

        #features in this frame
        f = features[features["frame"] == itime]  
        
        if f.empty:
            print(f"No features found for frame {itime}, skipping segmentation.")
            segments_all.append((itime, None, None))
            continue

        #perform segmentation
        segment_labels, segments = tobac.segmentation_2D(
            f,
            field_2d,
            dxy=dxy,
            threshold=norm_threshold,
            target="minimum"
        )
        #store results
        segments_all.append((itime, segment_labels, segments))
        
    #getting the list with number of images
    plot_frames = range(0, images_no)





    #===== PLOTTING =====

    #keep track of cells in previous frames for "gap_features_frames" frames
    cells_frames_before=[set() for _ in range(gap_features_frames)]
    #what cells are in the current frame
    cell_ids=set() 

    for i, itime in enumerate(plot_frames):

        #remove the oldest frame and add the current one
        cells_frames_before.pop(0)
        cells_frames_before.append(cell_ids)

        #get the cells in this frame
        cell_ids = set(trajectories[(trajectories["frame"] == itime)]["cell"].dropna().unique())

        #all_cells_in_gap collects all the cells in the previous gap_features_frames frames (a cell may have disappeared and reappeared)
        all_cells_in_gap=set()
        #map containing a list, for each cell in the current frame: all the frames it appeared in the previous gap_features_frames frames
        all_frames_for_cell = {}

        for j in range(gap_features_frames):
            #set sum
            all_cells_in_gap = all_cells_in_gap | cells_frames_before[j]
            for el in cells_frames_before[j]:
                if el not in all_frames_for_cell:
                    all_frames_for_cell[el] = []  #create a list for new cells
                #add the frame number where the cell appeared
                all_frames_for_cell[el].append(itime - (gap_features_frames - j))
    


        persisted = cell_ids & all_cells_in_gap   #intersection -> clouds present now and previously
        new_cells = cell_ids - all_cells_in_gap   #new this frame (may have reappeared after too long (> gap_features_frames))
        disappeared = all_cells_in_gap - cell_ids  #disappeared clouds in this frame

        original_img_name = os.path.splitext(os.path.basename(image_files[itime]))[0]
        
        #Get the field for this frame
        fig, axs = plt.subplots(figsize=(6, 6))
            
        smoothed_frame = ndimage.gaussian_filter(test_data_norm.isel(time=itime).values, sigma=smooth)

        temp_da = test_data_norm.isel(time=itime).copy()
        temp_da.data = smoothed_frame

        #consistent color range across all frames
        axs.imshow(temp_da.values, origin="upper", cmap="viridis")  # pixels are axes
        xlim = (0, temp_da.sizes['x'])
        ylim = (0, temp_da.sizes['y'])
    
        #forall cells in this frame plot the trajectory, center, segmentation and radius
        for cell_id in cell_ids:
            #get if the cell id is in the current frame
            track = trajectories[trajectories["cell"] == cell_id]
            f_weighted = track[(track["frame"] == itime)]
            
            printing_symbol=''
            color=''

            if(cell_id in new_cells):
                printing_symbol='^'
                color='white'
            else:
                printing_symbol='x'
                color='red'

            #print trajectory and center (if new or persisted the symbol changes)
            print_clouds_center_line(printing_symbol,color,f_weighted, itime, track, axs,cell_id, persisted,all_frames_for_cell)      
            
            if(len(f_weighted["x"])==0):
                continue

            #print cell id numbers on the plot for clarity
            print_cloud_labels(f_weighted, cell_id,xlim, ylim, axs)
            #print the radius
            add_circle_slice_filled(axs, f_weighted, radius=radius, xlim=xlim, ylim=ylim,color='red', alpha=0.05)


        #Extract segmentation for this frame from segments_all and print it            
        entry = next((s for s in segments_all if s[0] == itime), None)
        
        if entry is not None:
            _, seg_labels, _ = entry
            if seg_labels is not None:
                # seg_labels may have a single-element time dimension
                seg_labels2d = seg_labels.isel(time=0)  # drop the time dim for contour
                # Only plot contour if there are actual segmented pixels
                seg_labels2d.plot.contour(levels=[0.5], ax=axs, colors="k")
                


        #finalize the figure
        
        axs.set_title("")
        axs.set_xticks([])       #remove x-axis ticks
        axs.set_yticks([])       #remove y-axis ticks
        axs.set_xticklabels([])  #remove x-axis labels
        axs.set_yticklabels([])  #remove y-axis labels
        axs.axis('off')    
        out_path = os.path.join(output_folder, f"{original_img_name}.png")
        axs.set_xlim(0, temp_da.sizes["x"])
        axs.set_ylim(temp_da.sizes["y"], 0)  #since origin="upper"
        plt.savefig(out_path, dpi=150, bbox_inches="tight",pad_inches=0)
        plt.close(fig) 
        
def print_clouds_center_line(printing_symbol,color,f_weighted, itime, track, axs, cell_id,persisted_cells,all_frames_for_cell):
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
    cell_in_this_frame = not(track[track["frame"] == itime].empty)
 
    #if the cloud persisted from previous frames, print the last segment of the trajectory
    if(cell_id in persisted_cells and cell_in_this_frame):
        last_frame=(all_frames_for_cell[cell_id])[-1]
        line = track[(track["frame"] == last_frame) | (track["frame"] == itime)]  # last two frames

        #plot main trajectory (last step)
        axs.plot(
            line["x"],
            line["y"],
            color="blue",
            linewidth=1.5,
            alpha=0.5,
        )


    #plot trajectory with gradient (fading older traces)
    try:
        frames = all_frames_for_cell[int(cell_id)]
        #for all pair of frames
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
        #print("no frames for cell ", cell_id)

    #The cloud exists at this frame, mark the center 
    f_weighted.plot.scatter(
            x="x",
            y="y",
            s=40,
            ax=axs,
            color=color,
            marker=printing_symbol,
        )

def split_and_merge(trajectories,dxy,output_file):
    #just split for now
    d = tobac.merge_split.merge_split_MEST(trajectories, dxy=dxy)
    
    #convert to DataFrame
    df = d.to_dataframe().reset_index()

    #from the dataframe filter values (there are useless rows) and remove some duplicated columns
    filtered_df = df[
        (df["track"] == df["feature_parent_track_id"]) &
        (df["cell"] == df["feature_parent_cell_id"])
    ].drop(columns=["feature_parent_track_id", "feature_parent_cell_id","cell_child_feature_count","cell_ends_with_merge"])#add back cell_ends_with_merge

    #we split the data according to the different tracks (and remove the track column, which are now the keys)
    #groups is a map of track_id -> dataframe with the data for that track
    groups = {
        track_id: group.drop(columns=["track"])
        for track_id, group in filtered_df.groupby("track")
    }

    #the first frame in which each cell appears cell -> fame
    cell_first_frame = {}
    #the first frame in which each track appears track -> frame
    track_first_frame = {}

    #loop through rows of trajectories, collect first appearance of each cell
    for _, row in trajectories.iterrows():
        cell_id = row["cell"]
        frame = row["frame"]

        if cell_id not in cell_first_frame:
            cell_first_frame[cell_id] = frame

    #loop through groups to get first frame in which each track is born
    for track_id in groups.keys():
        if track_id not in track_first_frame:
            #loop through rows of groups[track_id], get the min cell
            g = groups[track_id]
            min_cell = g["cell"].min()
            track_first_frame[track_id] = cell_first_frame[min_cell]
    
    str_to_save = ""
    #loop through each group
    for track_id in groups.keys():
        str_to_save+=f"\nTrack {track_id}:"
        
        #get the dataframe
        g = groups[track_id]

        #cells_in_track: contains which cells are in the track, cell_id -> first frame in which it appears
        cells_in_track = {} 

        #loop through each row of that DataFrame
        for _, row in g.iterrows():
            #check if cells_in_track[row["cell"]] already exists
            cell_id=row["cell"]
            if cell_id in cells_in_track:
                continue
            #get the first frame in which this cell appears
            cells_in_track[cell_id] = cell_first_frame[cell_id]


        for keys in cells_in_track.keys():
            str_to_save+=f"\n  Cell ID: {keys} first appears in frame: {cells_in_track[keys]}"

        for keys in cells_in_track.keys():
            if(cells_in_track[keys] != track_first_frame[track_id]):
                parent_track_id = g.loc[g["cell"] == keys, "cell_parent_track_id"].iloc[0]
                str_to_save+=f"\n  --> Cell {keys} split (originated) from cell {parent_track_id} at frame {cells_in_track[keys]}"
    
    with open(output_file, "w") as f:
        f.write(str_to_save)

def print_cloud_labels(f_weighted, cell_id,xlim, ylim, axs):
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
    #get the position and adjust if too close to the edges
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
        x_pos -3,  #offset a bit to the right
        y_pos -3,  #offset upward slightly
        f"{int(cell_id)}",  #text = cloud id
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

    #get the position (center for the circle)
    cx = f_weighted["x"].iloc[0]
    cy = f_weighted["y"].iloc[0]

    #sample circle points
    theta = np.linspace(0, 2*np.pi, 300)
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    #clip points to the plot
    x_clipped = np.clip(x, xlim[0], xlim[1])
    y_clipped = np.clip(y, ylim[0], ylim[1])

    #polygon from clipped points
    polygon_points = np.column_stack([x_clipped, y_clipped])

    #add corners if any clipped point is at box edge
    corners = []
    if np.any(x < xlim[0]) and np.any(y < ylim[0]):
        corners.append([xlim[0], ylim[0]])
    if np.any(x > xlim[1]) and np.any(y < ylim[0]):
        corners.append([xlim[1], ylim[0]])
    if np.any(x < xlim[0]) and np.any(y > ylim[1]):
        corners.append([xlim[0], ylim[1]])
    if np.any(x > xlim[1]) and np.any(y > ylim[1]):
        corners.append([xlim[1], ylim[1]])

    #just plotting
    if corners:
        polygon_points = np.vstack([polygon_points, corners])

    polygon = patches.Polygon(polygon_points, closed=True,
                              facecolor=color, alpha=alpha,**kwargs)
    
    ax.add_patch(polygon)

    polygon_border = patches.Polygon(polygon_points, closed=True,
                              facecolor="none", alpha=0.3,edgecolor="red",linestyle="--",linewidth=1,**kwargs)
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
    
    #match pattern like cloud_200_YYYYMMDD_HHMM.png
    m = re.search(r'_(\d{8})_(\d+)\.png$', filename)
    if m:
        date = int(m.group(1))
        num = int(m.group(2))
        return (date, num)
    else:
        return (0, 0)




def run_tobac(inpu_folder, output_folder,lat_min,lat_max,lon_min,lon_max,n_min_threshold=0,smooth = 8):
    """
    The main function called from outside (main).
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
    locate_track(inpu_folder, output_folder,n_min_threshold,lat_min,lat_max,lon_min,lon_max,smooth)
    print("Locating & tracking procedure completed")
