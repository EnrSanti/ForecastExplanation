import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import defaultdict

#longmin longmax latmin latmax of FVG and Italy, longmin longmax, latmin latmax
coordinates=[11,15,44.5,48]
coordinates_italy=[6.5,18.5,36.5,48]
base_path = "../image_processing/fvg/output/"
locations_name_px_pos={}

'''
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
    global locations_name_px_pos

    locations=locations_name_px_pos
    # Unpack coordinates
    
    # Make sure output folder exists
    os.makedirs(image_output_folder, exist_ok=True)
    
    # Loop through all images
    for filename in sorted(os.listdir(image_input_folder)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue
        
        input_path = os.path.join(image_input_folder, filename)
        output_path = os.path.join(image_output_folder, filename)
        
        # Load image
        img = mpimg.imread(input_path)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img, origin="upper")
        
        # Plot each location as a red star
        for _, row in locations.iterrows():
            px,py=locations_name_px_pos[row["Location"]]
            ax.plot(px, py, marker=".", color="red", markersize=10)
            ax.text(px + 5, py - 5, row["Location"], color="red", fontsize=8)
        
        ax.axis("off")
        plt.tight_layout()
        
        # Save to output folder
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"Saved {output_path}")

def get_clouds_covering_locations(locations_name_px_pos, segment_labels_path="../image_processing/fvg/output/cloud_at_750m/segment_labels_all.npz"):
    
    """
    Returns a mapping: frame_index -> { cell_id -> [locations] }.
    """
    
    data = np.load(segment_labels_path)
    segment_labels_list = [data[key] for key in data][0:2]

    frame_cloud_map = {}
    print("shaaape ->", segment_labels_list[0].shape)
    for frame_idx, seg_labels in enumerate(segment_labels_list):
        # --- Handle non-2D arrays safely ---
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
        print("height % width:    ", height,"   ",width)
        clouds_to_locations = defaultdict(list)

        for loc_name, (px, py) in locations_name_px_pos.items():
            px_i = int(round(px))
            py_i = int(round(py))
            if 0 <= px_i < width and 0 <= py_i < height:
                cell_id = seg_labels[py_i, px_i]
                if cell_id > 0:
                    clouds_to_locations[int(cell_id)].append(loc_name)
            else:
                print(f"⚠️ Location {loc_name} outside frame {frame_idx} bounds (px={px_i}, py={py_i})")

        frame_cloud_map[frame_idx] = dict(clouds_to_locations)

    return frame_cloud_map

    



load_locations(coordinates,base_path)




#########################################################################

frame_cloud_map = get_clouds_covering_locations(locations_name_px_pos)

trajectories =  pd.read_csv(base_path+"cloud_at_750m/trajectories.csv")

# Display first 5 rows
print(trajectories.head())

#########################################################################



# Example: print clouds covering each location in frame 0
full_str=""
for key in frame_cloud_map.keys():
    for cell_id, locs in frame_cloud_map[key].items():
        full_str+=f"Frame {key} — Cloud {cell_id}: covers {locs}\n"
        print(f"[750m] Frame {key} — Cloud {cell_id}: covers {locs}")

with open("demofile.txt", "w") as f:
  f.write(full_str)
'''