import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import defaultdict
from datetime import datetime,timedelta

#longmin longmax latmin latmax of FVG and Italy, longmin longmax, latmin latmax
coordinates=[11,15,44.5,48]
coordinates_italy=[6.5,18.5,36.5,48]
base_path = "../image_processing/fvg/output/"
locations_name_px_pos={}

starting_date=None

def feature_to_cell_id(feature_id,trajectories):
    print("looking for feature ->",feature_id)
    feature_to_cell_map = trajectories.set_index("feature")["cell"].to_dict()
    print(feature_to_cell_map)
    return feature_to_cell_map.get(feature_id,None)

def geo_to_pixel(lon, lat, lon_min, lon_max, lat_min, lat_max, img_width, img_height):
    x = (lon - lon_min) / (lon_max - lon_min) * img_width
    y = (lat_max - lat) / (lat_max - lat_min) * img_height
    return x, y

def load_locations(coordinates,image_input_folder):
    global locations_name_px_pos
    locations_data = pd.read_csv("locations.csv")  # same folder or specify full path
    print(f"Loaded {len(locations_data)} locations.")

    lon_min, lon_max = coordinates[0], coordinates[1]
    lat_min, lat_max = coordinates[2], coordinates[3]

    #read a file just to get the image shape (they are all =)
    first_dir=sorted(os.listdir(image_input_folder))[0]
    first_dir=image_input_folder+first_dir
    first_file = sorted([f for f in os.listdir(first_dir) if os.path.isfile(os.path.join(first_dir, f)) and f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])[0]

    input_path = os.path.join(first_dir, first_file)
        
    # Load image
    img = mpimg.imread(input_path)
    height, width = img.shape[:2]
    print(type(locations_data))

    for _, row in locations_data.iterrows():
        px, py = geo_to_pixel(
            row["Long"], row["Lat"],
            lon_min, lon_max,
            lat_min, lat_max,
            width, height
        )
        locations_name_px_pos[row["Location"]]=(px,py)

def plot_locations_to_map(image_input_folder, image_output_folder, coordinates):
    # Load locations
    global locations_name_px_pos, starting_date

    locations=locations_name_px_pos
    # Unpack coordinates
    
    # Make sure output folder exists
    os.makedirs(image_output_folder, exist_ok=True)
    
    # Loop through all images
    for filename in sorted(os.listdir(image_input_folder)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue
        if(starting_date is None):
            parts = filename.split("_")
            date_str = parts[-2]         # '20191101'
            hour_str = parts[-1][:2]     # '05' → 5
    
            starting_date= datetime.strptime(date_str + hour_str, "%Y%m%d%H")

        input_path = os.path.join(image_input_folder, filename)
        output_path = os.path.join(image_output_folder, filename)
        
        # Load image
        img = mpimg.imread(input_path)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img, origin="upper")
        
        # Plot each location as a red star
        for location in locations.keys():
            px,py=locations[location]
            ax.plot(px, py, marker=".", color="red", markersize=10)
            ax.text(px + 5, py - 5, location, color="red", fontsize=8)
        
        ax.axis("off")
        plt.tight_layout()
        
        # Save to output folder
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"Saved {output_path}")

def get_clouds_covering_locations(locations_name_px_pos, segment_labels_path,trajectories):
    
    """
    Returns a mapping: frame_index -> { cell_id -> [locations] }.
    """
    
    data = np.load(segment_labels_path,allow_pickle=True)
    print("loading ",segment_labels_path)
   
    segment_labels_list = [data[key] for key in data]

    frame_cloud_map = {}


    for frame_idx, seg_labels in enumerate(segment_labels_list):
        # --- Handle non-2D arrays safely ---
        try: #to deal with empty frames (i.e. frames without blobs)
            if seg_labels.ndim == 3:
                # Common cases: (y, x, 1) or (1, y, x)
                if seg_labels.shape[-1] == 1:
                    seg_labels = seg_labels[..., 0]
                elif seg_labels.shape[0] == 1:
                    seg_labels = seg_labels[0]
                else:
                    raise ValueError(f"Unexpected shape {seg_labels.shape} for frame {frame_idx}")
            elif seg_labels.ndim != 2:
                raise ValueError(f"Unsupported number of dimensions: {seg_labels.ndim}")

            height, width = seg_labels.shape
            
            clouds_to_locations = defaultdict(list)

            for loc_name, (px, py) in locations_name_px_pos.items():
                px_i = int(round(px))
                py_i = int(round(py))
                if 0 <= px_i < width and 0 <= py_i < height:
                    feature_id = seg_labels[py_i, px_i]
                    if(feature_id>0):
                        cell_id=feature_to_cell_id(int(feature_id),trajectories)
                        
                        clouds_to_locations[int(cell_id)].append(loc_name)
                else:
                    print(f"⚠️ Location {loc_name} outside frame {frame_idx} bounds (px={px_i}, py={py_i})")

            frame_cloud_map[frame_idx] = dict(clouds_to_locations)
        except:
            frame_cloud_map[frame_idx]={}

  
    
    return frame_cloud_map

def frame_index_to_timestamp(index,starting_date,frame_time_interval):
    """
    Calculate the timestamp of a frame.
    
    Parameters:
        index (int): frame index (>=0)
        starting_date (datetime): datetime of frame 0
        frame_time_interval (timedelta): time between two consecutive frames
        
    Returns:
        datetime: timestamp of the frame at the given index
    """
    if index < 0:
        raise ValueError("Frame index must be >= 0")
    
    # Convert integer hours to timedelta if necessary
    if isinstance(frame_time_interval, int):
        frame_time_interval = timedelta(hours=frame_time_interval)
    
    return starting_date + index * frame_time_interval


    


load_locations(coordinates,base_path)

for entry in os.listdir(base_path):
    full_path = os.path.join(base_path, entry)
    if os.path.isdir(full_path):
        #later this first stage can be skipped
        plot_locations_to_map(full_path,"examples_from_blobs/"+entry,coordinates)

        #########################################################################
        #compute the map that for each cloud tells me which locations I am covering
        trajectories =  pd.read_csv(full_path+"/trajectories.csv")
        frame_cloud_map = get_clouds_covering_locations(locations_name_px_pos,full_path+"/segment_labels_all.npz",trajectories)
        
        #########################################################################


        # Example: print clouds covering each location in frame 0
        full_str="starting date: "+str(starting_date)+"\n"
        for key in frame_cloud_map.keys():
            for cell_id, locs in frame_cloud_map[key].items():
                time=frame_index_to_timestamp(key,starting_date,1) #1h between each frame
                full_str+=f"[{full_path.rsplit('/', 1)[-1]}] -- at {str(time)} -- Cloud {cell_id}: covers {locs}\n"
                print(f"[{full_path.rsplit('/', 1)[-1]}] -- at {str(time)} -- Cloud {cell_id}: covers {locs}\n")

        with open(f"{"examples_from_blobs/"+full_path.rsplit('/', 1)[-1]}/clouds_covering.txt", "w") as f:
            f.write(full_str)
        frame_cloud_map={}
