import numpy as np
from reasoning.generate_examples import load_locations
import pandas as pd
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from PIL import Image
from reasoning.generate_examples import folder_types, folders_suff, frame_index_to_timestamp

import os
from collections import defaultdict
import math

coordinates=[]
base_path=""
locations_pos_px={}
city_map={}


def plot_city_connections_on_image(image_path, city_locs, adjacency, save_path=None, linewidth=2):
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open(image_path)
    W, H = img.size

    # prepare axis with matching coordinates
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)

    # display image exactly at pixel coords
    ax.imshow(img, extent=[0, W, H, 0])     # IMPORTANT

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                       # IMPORTANT
    ax.set_aspect('equal')                  # IMPORTANT
    ax.set_axis_off()

    # plot edges
    for city, neighbors in adjacency.items():
        x1, y1 = city_locs[city]
        for nb in neighbors:
            x2, y2 = city_locs[nb]
            ax.plot([x1, x2], [y1, y2], '-', color='white', linewidth=linewidth)

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()


def build_delaunay_adjacency_filtered(city_locs, length_factor=1.8):
    """
    Build a filtered Delaunay adjacency graph.

    Long edges (outliers) are removed using a threshold:
        L(edge) <= length_factor * median_edge_length

    Parameters:
        city_locs : dict {city: (x,y)} or DataFrame with index=city, columns=[x,y]
        length_factor : multiplier for filtering (typical 1.5–2.5)

    Returns:
        adjacency : dict {city_name: set(neighbor_cities)}
    """

    # -------------------------------------------------------
    # 1. Convert input to consistent arrays
    # -------------------------------------------------------
    if isinstance(city_locs, pd.DataFrame):
        coords = city_locs.values
        names = list(city_locs.index)
    else:
        names = list(city_locs.keys())
        coords = np.array([city_locs[name] for name in names], dtype=float)

    coords = np.asarray(coords, dtype=float)

    # -------------------------------------------------------
    # 2. Delaunay triangulation
    # -------------------------------------------------------
    tri = Delaunay(coords)

    # Collect all Delaunay edges (undirected)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a = simplex[i]
            b = simplex[(i+1) % 3]
            edge = tuple(sorted((a, b)))
            edges.add(edge)

    # -------------------------------------------------------
    # 3. Compute lengths for filtering
    # -------------------------------------------------------
    lengths = np.array([
        np.linalg.norm(coords[a] - coords[b])
        for a, b in edges
    ])

    median_len = np.median(lengths)
    max_len = median_len * length_factor

    # -------------------------------------------------------
    # 4. Build adjacency with filtering
    # -------------------------------------------------------
    adjacency = {name: set() for name in names}

    for (a, b), L in zip(edges, lengths):
        if L <= max_len:  # keep only "reasonable" edges
            city_a = names[a]
            city_b = names[b]
            adjacency[city_a].add(city_b)
            adjacency[city_b].add(city_a)

    return adjacency


def build_city_adjacency_graph_for_fronts(base_path, coordinates):
    
    path="./reasoning/screen.png"
    # 2. Get the neighborhoods
    locations_name_px_pos = load_locations(coordinates,base_path)
    adj = build_delaunay_adjacency_filtered(locations_name_px_pos)  # a = your DataFrame/dict

    for city, neighbors in adj.items():
        print(f"{city} → {sorted(neighbors)}")

    '''
    plot_city_connections_on_image(
        image_path=path,
        city_locs=locations_name_px_pos,
        adjacency=adj,
        save_path="triangulation_overlay.png"
    )
    '''
    return adj, locations_name_px_pos

def init_fronts_generation(path,_coordinates):
    global coordinates
    global base_path
    global city_map
    global locations_pos_px
    base_path=path
    coordinates=_coordinates
    city_map, locations_pos_px = build_city_adjacency_graph_for_fronts(path, coordinates)




# Assume these are globally available, set by init_fronts_generation
# global coordinates
# global base_path
# global map # Adjacency graph (city -> list of adjacent cities)
# global locations_pos_px # Dictionary: location_name -> (px, py)
# global starting_date # Assume this is available for timestamp conversion
# global feature_to_cell_id # Function to map segment ID to cell ID (from trajectories)
# global frame_index_to_timestamp # Function to convert frame index to yyyy,mm,dd,h

# Constants for the front generation
PIXEL_STEP_SIZE = 5 # Step size for checking pixels on the line segment

def get_pixel_line(p1_px, p1_py, p2_px, p2_py):
    """
    Generates pixel coordinates along the line segment between (p1_px, p1_py) and (p2_px, p2_py).
    Uses a simple line algorithm, sampling every PIXEL_STEP_SIZE.
    """
    points = []
    dx = p2_px - p1_px
    dy = p2_py - p1_py
    distance = math.sqrt(dx**2 + dy**2)
    
    if distance == 0:
        return []
    
    # Number of steps to check
    steps = int(distance / PIXEL_STEP_SIZE)
    if steps < 1: # Ensure at least one check if cities are close but not same pixel
        steps = 1
        
    for i in range(1, steps):
        t = i / steps
        px = int(round(p1_px + t * dx))
        py = int(round(p1_py + t * dy))
        points.append((px, py))
        
    return points

def is_line_clear(p1_name, p2_name, seg_labels, locations_name_px_pos):
    """
    Checks if there are any segmented features (seg_labels > 0) 
    on the straight line between the two cities.
    
    Returns True if the line is clear (no seg_labels > 0), False otherwise.
    """
    (p1_px, p1_py) = locations_name_px_pos[p1_name]
    (p2_px, p2_py) = locations_name_px_pos[p2_name]
    
    height, width = seg_labels.shape


    #with a single cluster per image always clear
    '''
    line_pixels = get_pixel_line(p1_px, p1_py, p2_px, p2_py)
    
    for px, py in line_pixels:
        if 0 <= px < width and 0 <= py < height:
            if seg_labels[py, px] > 0:
                return False # Found a segment feature on the line
    '''
    return True # Line is clear

def process_segmentation_data(segment_labels_path):
    """Loads and normalizes the segment labels from an .npz file."""
    try:
        data = np.load(segment_labels_path, allow_pickle=True)
        segment_labels_list = [data[key] for key in data if data[key].size > 0]
    except:
        return []

    processed_list = []
    for seg_labels in segment_labels_list:
        try:
            # Normalize 3D to 2D
            if seg_labels.ndim == 3:
                if seg_labels.shape[-1] == 1:
                    seg_labels = seg_labels[..., 0]
                elif seg_labels.shape[0] == 1:
                    seg_labels = seg_labels[0]
                else:
                    # Treat unexpected 3D as error, or decide a standard way to collapse
                    seg_labels = seg_labels.mean(axis=0).astype(int) 
            
            if seg_labels.ndim == 2:
                processed_list.append(seg_labels)
            else:
                processed_list.append(np.array([])) # Placeholder for invalid frames

        except Exception as e:
            processed_list.append(np.array([]))
            
    return processed_list

def get_location_segment_id(loc_name, seg_labels, locations_name_px_pos):
    """
    Returns the segment ID (or 0 if none) covering a location.
    Handles location out of bounds by returning 0.
    """
    if seg_labels.size == 0:
        return 0
        
    (px, py) = locations_name_px_pos[loc_name]
    px_i = int(round(px))
    py_i = int(round(py))
    height, width = seg_labels.shape
    
    if 0 <= px_i < width and 0 <= py_i < height:
        return seg_labels[py_i, px_i]
    return 0

def generate_fronts_hum(starting_date):
    """
    Generates facts about humidity fronts between adjacent cities 
    based on segmentation data.
    """
    global base_path 
    global city_map
    global locations_pos_px

    map = city_map
        
    # Assuming 'humidity' is index 1
    target_type = folder_types[1] 


    
    all_fronts_facts = []

    for level, suff in folders_suff.items():
        folder_name = target_type + suff
        full_path = os.path.join(base_path, folder_name)

        if os.path.isdir(full_path):
            
            print(f"Processing folder: {folder_name} for humidity fronts.")
            
            # Load all segment labels for this altitude
            segment_labels_path = os.path.join(full_path, "segment_labels_all.npz")
            segment_labels_list = process_segmentation_data(segment_labels_path)
            
            # Trajectories are not needed for *front* detection, only for *ID* tracking, 
            # but since we only check *presence* (ID > 0), we can skip loading trajectories.
            
            altitude_facts = []
            

            for frame_idx, seg_labels in enumerate(segment_labels_list):

                yyyy, mm, dd, h = frame_index_to_timestamp(frame_idx, starting_date, 1) 
                
                 # Iterate over all adjacent city pairs
                for city1, adjacents in map.items():
                    for city2 in adjacents:
                        # Ensure we only check each pair once (e.g., A->B, but skip B->A if already checked)
                        if city1 > city2: 
                            continue 

                        
                        # 1. Check feature coverage for both cities
                        city1_seg_id = get_location_segment_id(city1, seg_labels, locations_pos_px)
                        city2_seg_id = get_location_segment_id(city2, seg_labels, locations_pos_px)
                        
                        # Front condition: one covered (seg_id > 0), one not (seg_id == 0)
                        is_front_candidate = (city1_seg_id != city2_seg_id == 0)


                        if is_front_candidate:
                            # 2. Check for clear line segment
                            if is_line_clear(city1, city2, seg_labels, locations_pos_px):
                                
                                front_pred = f"hum_front_{level}_hPa({city1},{city2},{yyyy},{mm},{dd},{h})."
                                altitude_facts.append(front_pred)
                                # If one is covered and the other is not, the 'front' is between them
                                

            # Write facts to a file
            output_dir = f"reasoning/humidity/{folder_name}"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "humidity_fronts.txt")
            
            full_str = f"% format: hum_front_[alt]_hPa(city1,city2,yyyy,mm,dd,h).\n\n"
            full_str += "\n".join(altitude_facts)
            
            with open(output_file, "w") as f:
                f.write(full_str)
                f.write("\n")
                
            all_fronts_facts.extend(altitude_facts)
            print(f"Finished processing {folder_name}. Saved {len(altitude_facts)} facts.")

    return all_fronts_facts

def generate_fronts_temp(starting_date):

    """
    Generates facts about humidity fronts between adjacent cities 
    based on segmentation data.
    """
    global base_path 
    global city_map
    global locations_pos_px

    map = city_map
        
    # Assuming 'humidity' is index 1
    target_type = folder_types[2] 

    all_fronts_facts = []

    for level, suff in folders_suff.items():
        folder_name = target_type + suff
        full_path = os.path.join(base_path, folder_name)

        if os.path.isdir(full_path):
            
            print(f"Processing folder: {folder_name} for temperature fronts.")
            
            # Load all segment labels for this altitude
            segment_labels_path = os.path.join(full_path, "segment_labels_all.npz")
            segment_labels_list = process_segmentation_data(segment_labels_path)
            
            # Trajectories are not needed for *front* detection, only for *ID* tracking, 
            # but since we only check *presence* (ID > 0), we can skip loading trajectories.
            
            altitude_facts = []
            

            for frame_idx, seg_labels in enumerate(segment_labels_list):

                yyyy, mm, dd, h = frame_index_to_timestamp(frame_idx, starting_date, 1) 
                
                 # Iterate over all adjacent city pairs
                for city1, adjacents in map.items():
                    for city2 in adjacents:
                        # Ensure we only check each pair once (e.g., A->B, but skip B->A if already checked)
                        if city1 > city2: 
                            continue 

                        
                        # 1. Check feature coverage for both cities
                        city1_seg_id = get_location_segment_id(city1, seg_labels, locations_pos_px)
                        city2_seg_id = get_location_segment_id(city2, seg_labels, locations_pos_px)
                        
                        # Front condition: one covered (seg_id > 0), one not (seg_id == 0)
                        is_front_candidate = (city1_seg_id != city2_seg_id == 0)


                        if is_front_candidate:
                            # 2. Check for clear line segment
                            if is_line_clear(city1, city2, seg_labels, locations_pos_px):
                                
                                front_pred = f"temp_front_{level}_hPa({city1},{city2},{yyyy},{mm},{dd},{h})."
                                altitude_facts.append(front_pred)
                                # If one is covered and the other is not, the 'front' is between them
                                

            # Write facts to a file
            output_dir = f"reasoning/temp/{folder_name}"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "temp_fronts.txt")
            
            full_str = f"% format: temp_front_[alt]_hPa(city1,city2,yyyy,mm,dd,h).\n\n"
            full_str += "\n".join(altitude_facts)
            
            with open(output_file, "w") as f:
                f.write(full_str)
                f.write("\n")
                
            all_fronts_facts.extend(altitude_facts)
            print(f"Finished processing {folder_name}. Saved {len(altitude_facts)} facts.")

    return all_fronts_facts
    