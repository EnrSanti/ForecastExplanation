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
import cv2
from collections import defaultdict
from itertools import combinations
import uuid
from typing import Dict, List, Set,Union
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.segmentation import mark_boundaries

def select_indices_best_match(overlap_percentages, masses, threshold, X):

    """
    Select the subset of regions whose overlap >= threshold, such that their total mass
    is closest to a target mass X (can go over or under).

    Args:
        overlap_percentages (list[float]): Overlap percentages of each region.
        masses (list[float]): Mass or size of each region.
        threshold (float): Minimum overlap percentage to consider.
        X (float): Target total mass.

    Returns:
        list[int]: Indices of the subset closest to X.
    """
    
    # Step 1: filter indices where overlap >= threshold
    valid_indices = [i for i, o in enumerate(overlap_percentages) if o >= threshold]
    if not valid_indices:
        return []

    valid_masses = [masses[i] for i in valid_indices]
    
    best_diff = float('inf')
    best_subset = []

    n = len(valid_indices)
    # Brute-force search over all combinations
    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            total = sum(valid_masses[j] for j in combo)
            diff = abs(total - X)
            if diff < best_diff:
                best_diff = diff
                best_subset = combo

    # Convert back to original indices
    return [valid_indices[j] for j in best_subset]

def find_extended_overlap_blobs_inferred(
    segment_labels: np.ndarray, 
    trajectories: pd.DataFrame, 
    border_thickness_px: int
) -> Dict[Union[int, str], List[Union[int, str]]]:
    """
    Checks for intersection between the EXTENDED (dilated) areas of all segmentations 
    by inferring the frame index from the segment_labels data.
    """
    if segment_labels.ndim != 2:
        raise ValueError("segment_labels must be 2D (y, x).")

    # --- PART 1: INFER FRAME INDEX AND SETUP MAPPING ---

    all_segment_ids = np.unique(segment_labels)
    feature_ids_in_field = [int(i) for i in all_segment_ids if i > 0]
    
    if not feature_ids_in_field:
        return {} # No features to process

    # Find a sample feature ID present in the segmentation field
    sample_feature_id = feature_ids_in_field[0]
    
    # Infer the frame index from the trajectories using the sample feature ID
    try:
        frame_index = trajectories[trajectories['feature'] == sample_feature_id]['frame'].iloc[0]
    except IndexError:
        raise ValueError(
            f"Feature ID {sample_feature_id} from segment_labels not found in trajectories. "
            "Ensure segment_labels correspond to a frame linked in trajectories."
        )

    # Filter trajectories for the current frame
    current_frame_features = trajectories[trajectories["frame"] == frame_index]
    
    # Create necessary mappings and lists
    feature_to_cell_map = current_frame_features.set_index("feature")["cell"].to_dict()
    feature_ids = [fid for fid in feature_ids_in_field if fid in feature_to_cell_map]
    cell_ids_in_frame = current_frame_features["cell"].unique()
    
    overlap_map: Dict[Union[int, str], Set[Union[int, str]]] = {
        int(cell_id): set() for cell_id in cell_ids_in_frame
    }
    
    # --- PART 2: PREPARE THE MASTER EXTENDED LABEL FIELD (L_ext) ---

    all_features_mask = segment_labels == 0 
    distances, indices = ndimage.distance_transform_edt(
        all_features_mask, 
        return_indices=True
    )
    
    L_nearest_id = segment_labels[tuple(indices)]
    L_ext = np.where(distances <= border_thickness_px, L_nearest_id, 0)
    
    # Dilation kernel needed only for A_ext creation inside the loop
    dilation_kernel_size = 2 * border_thickness_px + 1
    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), dtype=bool)
    
    # --- PART 3: ITERATE AND FIND OVERLAP (A_ext intersect L_ext) ---
    
    for current_id in feature_ids:

        current_cell_id = feature_to_cell_map.get(current_id)
        
        current_cell_id=int(current_cell_id)
        
            
        # 1. Create A_ext
        current_feature_mask = (segment_labels == current_id)
        extended_area_mask = ndimage.binary_dilation(
            current_feature_mask, 
            structure=dilation_kernel
        )
        
        # 2. Find all IDs present in the intersection of A_ext and L_ext
        segment_ids_in_extended_area = extended_area_mask * L_ext
        overlapping_ids_list = np.unique(segment_ids_in_extended_area)
        
        # 3. Translate and Record Overlap
        for overlap_id_raw in overlapping_ids_list:
            overlap_id = int(overlap_id_raw)
            
            if overlap_id > 0 and overlap_id != current_id:
                
                cell_id_overlap = feature_to_cell_map.get(overlap_id)
                
                if cell_id_overlap is not None and current_cell_id != cell_id_overlap:
                    
                    overlap_map.setdefault(current_cell_id, set()).add(cell_id_overlap)
                    overlap_map.setdefault(cell_id_overlap, set()).add(current_cell_id)

    # Finalize the output
    final_overlap_map = {
        cell_id: sorted(list(s)) for cell_id, s in overlap_map.items() if s
    }
    
    return final_overlap_map

def plot_feature_borders(segment_labels: np.ndarray, ax: plt.Axes, border_thickness_px: int = 1,  border_color: str = "red", segments_to_plot: set = None):


    """
    Adds a border around the segmented features to a plot.

    Args:
        segment_labels (np.ndarray): The integer-labeled segmentation field
                                     (output of tobac.segmentation_2D).
        ax (plt.Axes): The Matplotlib axes object to plot the border onto.
        border_thickness_px (int): The thickness of the border in pixels.
                                   (Approximated by dilation/erosion).
        border_color (str): The color for the border.
        segments_to_plot (set, optional): A set of segment labels (integers)
                                          to include. If None, all are plotted.
    """
    # 1. Create a binary mask of the features to be plotted
    if segments_to_plot is not None:
        # Create a mask only for the specified segment IDs
        mask = np.isin(segment_labels, list(segments_to_plot))
        # Zero out the other segment labels
        temp_labels = np.where(mask, segment_labels, 0)
    else:
        temp_labels = segment_labels

    # 2. Use a boundary marking function to get the border mask
    # The mark_boundaries function returns an RGB image where boundaries are colored.
    # We only need the mask of the boundary pixels.
    # Note: mark_boundaries will plot white boundaries on a black background
    # if the input image is also all black.
    
    # Create a small identity image for mark_boundaries to work with
    # The boundaries will be colored red on this output, based on temp_labels.
    identity_image = np.zeros_like(temp_labels, dtype=np.uint8)
    
    # Mark boundaries on the identity image
    boundary_image = mark_boundaries(
        identity_image,
        temp_labels,
        color=(1, 1, 1)  # Use white (1, 1, 1) for the border color in the mask
    )
    
    # Convert the boundary-marked image to a simple binary mask
    # A pixel is part of the border if any of the RGB channels are 1 (white)
    border_mask = np.any(boundary_image == 1, axis=-1)

    # 3. Dilate the border to achieve the desired thickness (m pixels)
    # The border_mask currently has a thickness of 1 pixel.
    # We use dilation to make it thicker.
    # The structure/kernel size should be (2*m + 1) to get thickness m.
    # For m=1, this is 3x3. For m=2, this is 5x5, etc.
    dilation_kernel_size = 2 * border_thickness_px + 1
    dilation_kernel = np.ones(
        (dilation_kernel_size, dilation_kernel_size), dtype=bool
    )
    
    # Dilate the mask
    thick_border_mask = ndimage.binary_dilation(border_mask, structure=dilation_kernel)

    # 4. Plot the thick border mask as a contour on the axes
    # We create a simple Xarray DataArray for easy plotting with xarray's .plot()
    # Assuming segment_labels has the same spatial dimensions as the original data
    
    # Find coordinates from the data already on the axes (if available)
    # A simpler approach is to use standard Matplotlib's contourf
    
    # Create a mask for plotting only the border, not the feature itself
    # Subtract the original feature area from the dilated border area
    feature_area_mask = temp_labels > 0
    final_border_mask = np.logical_and(thick_border_mask, ~feature_area_mask)
    
    # Use Matplotlib's contour to plot the boundary of the final_border_mask
    # We plot the '0.5' level of the binary mask.
    # You might need to interpolate this mask onto the original grid for
    # a smooth plot if using xarray's plot method, but contour works well.

    # We can plot the mask itself with pcolormesh or contourf for a filled border
    # Get the extent/coordinates from the original plot if possible, or use indices
    
    # Use imshow/pcolormesh with an alpha channel to overlay the colored border
    # Create an RGBA array for overlay
    ny, nx = final_border_mask.shape
    rgba_border = np.zeros((ny, nx, 4))
    
    # Set the color and full opacity for the border pixels
    # We use a color conversion here if the color is a string (e.g., 'red')
    # Using red for simplicity (R=1, G=0, B=0)
    # The alpha channel is 1 for border pixels, 0 otherwise
    if border_color == "red":
        rgba_border[final_border_mask] = [1.0, 0.0, 0.0, 1.0]
    # Add other colors as needed, or use matplotlib.colors.to_rgba

    # Overlay the image onto the existing plot.
    # Since your x-axis/y-axis are likely coordinates, you may need to
    # adjust the extent, but using Matplotlib's default index plotting
    # on top of xarray's plot should generally align.
    ax.imshow(
        rgba_border,
        origin="upper", # Adjust based on how your original data was plotted
        extent=ax.get_xlim() + ax.get_ylim(), # Match existing plot's extent
        alpha=0.4, # Alpha for the whole layer
        zorder=5 # Ensure border is on top
    )
                                                                       
def get_splits(overlap_map, trajectories, time_step,frames_no,gap_frames,segment_labels_current, segment_labels_prev,new_born_curr):
    #get the cells for the next time step
    print("get_splits called")

    frames_considered = 0
    cells_prev_step = []
    prev_time_step = time_step - 1
    for time in range(prev_time_step,time_step):
        print("considering frames from ",time," to ",time_step)
        frames_considered+=1
        cells_prev_step.append(trajectories[trajectories["frame"] == time]["cell"].unique())


    current_frame_features = trajectories[trajectories["frame"] == time_step]
    # Create necessary mappings and lists
    cell_ids_in_frame = current_frame_features["cell"].unique()

    if(frames_considered==0):
        return 

    
    #for frame in range(frames_considered):
    #map_areas_current=get_segment_areas_px(segment_labels.isel(time=time_step).values,trajectories)
    
    map_areas_curr=get_segment_areas_px(segment_labels_current.isel(time=0).values,trajectories)
    map_areas_prev=get_segment_areas_px(segment_labels_prev.isel(time=0).values,trajectories)

    prev_frame_features = trajectories[trajectories["frame"] == prev_time_step]
    # Create necessary mappings and lists
    cell_ids_in_prev_frame = prev_frame_features["cell"].unique()
    #cast to ints cell_ids_in_next_frame
    cell_ids_in_prev_frame = [int(cid) for cid in cell_ids_in_prev_frame]
    cell_ids_in_frame = [int(cid) for cid in cell_ids_in_frame]

    try:
    
        return detect_splits_by_area(overlap_map,map_areas_curr,map_areas_prev, cell_ids_in_frame,cell_ids_in_prev_frame,0.6,new_born_curr,time_step,segment_labels_current.isel(time=0).values, segment_labels_prev.isel(time=0).values,trajectories)
    except:
        return ""
def detect_splits_by_area(
    overlap_map,
    map_areas_curr,
    map_areas_prev,
    cell_ids_curr,
    cell_ids_prev,
    area_ratio_threshold,
    new_born_curr,
    frame_no,
    segment_labels_current, 
    segment_labels_prev,
    trajectories,
):
    print(overlap_map)

    """
    Detects 1 -> N cloud splits events between two consecutive frames.

    Args:
        overlap_map (dict[int, list[int]]): Overlap relations between current-frame cells.
        map_areas_curr (dict[int, float]): Area (or mass) of each cell in current frame.
        map_areas_next (dict[int, float]): Area (or mass) of each cell in next frame.
        cell_ids_curr (list[int]): Cell IDs present in current frame.
        cell_ids_next (list[int]): Cell IDs present in next frame.
        area_ratio_threshold (float): Minimum ratio sum(area_i)/area_next to confirm merge.

    Returns:
        list[dict]: List of detected merges with details.
    """

    splits = []
    already_split = set()
    splits_at_frame = ""
    for cell_id in cell_ids_prev:
        #può essersi splittato mantenendo o meno se stesso
        
        mass_previous_split=map_areas_prev.get(cell_id,0)
        #overlapping_with = overlap_map.get[cell_id]
        
            

        #gli altri candidati devono essere newborn e confinanti
        candidates = [
            int(c) for c in new_born_curr if c not in already_split
        ]

        #if the cell also exists now we just look for the remaining mass
        if(cell_id in map_areas_curr and cell_id not in candidates):
            candidates.append(cell_id)
        if len(candidates) == 0:
            continue

        overlap_percentage=[]
        mass=[]
        #do un check ai confinanti dei confianti SE SONO NEWBORN SE AGGIUNGONO UN PO ALLA MASSA e se intersection_next_frame è alta 

        print("candidates -> -> ",candidates)

        for c in candidates:
            if(c not in map_areas_curr): #SI SERVE. don't touch
                continue
            overlap_percentage.append(intersection_next_frame(c,cell_id, segment_labels_current, segment_labels_prev,trajectories,frame_no))
            mass.append(int(map_areas_curr[c]))


        print("overlap % ->",overlap_percentage)
        print("mass -> ",mass)
        print("mass to match -> ",mass_previous_split)
        indexes = select_indices_best_match(overlap_percentage,mass,area_ratio_threshold,mass_previous_split)
        print("indexes   ->     ",indexes)
        
        for index in indexes:
            already_split.add(candidates[index])
            if(candidates[index]!=cell_id):
                splits_at_frame += f"Cell {cell_id} split into {candidates[index]} at frame {frame_no}\n"
            elif(len(indexes)>1):
                print(indexes)
                print(candidates)
                splits_at_frame += f"Cell {cell_id} split but also remained at frame {frame_no}\n"
            
    return splits_at_frame    
        
def features_to_cell_ids(unique_labels_ids,trajectories):
    feature_to_cell_map = trajectories.set_index("feature")["cell"].to_dict()
    cell_ids = []
    for fid in unique_labels_ids:
        cell_id = feature_to_cell_map.get(fid)
        if cell_id is not None:
            cell_ids.append(int(cell_id))
    return cell_ids

def feature_to_cell_id(feature_id,trajectories):
    feature_to_cell_map = trajectories.set_index("feature")["cell"].to_dict()
    return feature_to_cell_map.get(feature_id,None)

def feature_from_cell_id(cell_id, trajectories, frame):
    """
    Return the single feature_id belonging to the given cell_id at a specific frame.
    If no feature is found, returns None.
    """
    subset = trajectories[
        (trajectories["cell"] == cell_id) &
        (trajectories["frame"] == frame)
    ]

    if subset.empty:
        print(f"No feature found for cell {cell_id} at frame {frame}.")
        return None

    feature_id = int(subset["feature"].iloc[0])
    #print(f"Cell {cell_id} at frame {frame} -> Feature {feature_id}")
    return feature_id

                             #nuova     #old      #dove cerco 1           #dove cerco 2 

def intersection_next_frame(cell_id_1, cell_id_2, segment_labels_current, segment_labels_prev,trajectories,curr_time):
    
    print("checking intersection between cell ",cell_id_1, "frame ",curr_time," and cell",cell_id_2," at frame ",curr_time-1)
    # --- Apply label transformations if provided ---
    seg_curr = np.copy(segment_labels_current)
    seg_prev = np.copy(segment_labels_prev)


    #OLD DERIVA DAL CELL TO FEATURE AT A SPECIFIC FRAME


    cell_1_feature=feature_from_cell_id(cell_id_1,trajectories,curr_time)
    cell_2_feature=feature_from_cell_id(cell_id_2,trajectories,curr_time-1)
    seg_curr[seg_curr != cell_1_feature] = 0
    seg_prev[seg_prev != cell_2_feature] = 0



    mask_1 = (seg_curr == cell_1_feature)
    mask_2 = (seg_prev == cell_2_feature)

    # Areas of each cloud
    area_1 = np.sum(mask_1)
    area_2 = np.sum(mask_2)
  
    if area_1 == 0 or area_2 == 0:
        return 0.0

    # Intersection area
    intersection = np.logical_and(mask_1, mask_2)
    intersection_area = np.sum(intersection)

    if intersection_area == 0:
        return 0.0

    # Normalize by the smaller cloud
    smaller_area = min(area_1, area_2)
    overlap_ratio = intersection_area / smaller_area

    # Clip to [0,1] to avoid floating precision issues
    return float(np.clip(overlap_ratio, 0.0, 1.0))
    
def get_segment_areas_px(segment_labels_2d,trajectories):
    """
    Calculates the area (in pixels) for each unique segment in a 2D segment label array.

    Args:
        segment_labels_2d (np.ndarray): The 2D segmentation label array 
                                        (e.g., segment_labels.isel(time=0).values).

    Returns:
        Dict[int, int]: A dictionary mapping Segment ID (Feature ID) to Area (in pixels).
    """
    # 1. Flatten the array and count the occurrences of each unique integer label
    unique_labels, counts = np.unique(segment_labels_2d, return_counts=True)
    
    #remove 0 (bg value):
    non_zero_mask = (unique_labels != 0)

    # 2. Apply the mask to both arrays
    unique_labels = unique_labels[non_zero_mask]
    counts = counts[non_zero_mask]

    cells=features_to_cell_ids(unique_labels,trajectories)
    area_map ={}
    i=0
    for cell in cells:
        area_map[cell]=counts[i]
        i+=1
    
    return area_map