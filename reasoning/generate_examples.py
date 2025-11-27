import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from PIL import Image
import re

#longmin longmax latmin latmax of FVG and Italy, longmin longmax, latmin latmax
coordinates=[11,15,44.5,48]
coordinates_italy=[6.5,18.5,36.5,48]
folder_types = ["cloud","humidity","temp"]

folders_suff = {
    1000: "_at_100m", 
    925: "_at_750m",
    850: "_at_1_4km",
    700: "_at_3km", 
    500: "_at_5_5km", 
    300: "_at_9km"
}
locations_name_px_pos={}

fact_pattern = re.compile(
    r'(?P<pred>forecasted_rain|forecasted_sky)\('
    r'\s*(?P<city>[a-zA-Z_]+)\s*,\s*'
    r'(?P<value>[^)]+)\)'
)

starting_date=None

def feature_to_cell_id(feature_id,trajectories):
    #print("looking for feature ->",feature_id)
    feature_to_cell_map = trajectories.set_index("feature")["cell"].to_dict()
    return feature_to_cell_map.get(feature_id,None)

def geo_to_pixel(lon, lat, lon_min, lon_max, lat_min, lat_max, img_width, img_height):
    x = (lon - lon_min) / (lon_max - lon_min) * img_width
    y = (lat_max - lat) / (lat_max - lat_min) * img_height
    return x, y

def load_locations(coordinates,image_input_folder):
    global locations_name_px_pos
    locations_data = pd.read_csv("reasoning/locations.csv")  # same folder or specify full path
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
    return locations_name_px_pos

def get_starting_date(filename):
    global starting_date
    if(starting_date is None):
        parts = filename.split("_")
        date_str = parts[-2]         # '20191101'
        hour_str = parts[-1][:2]     # '05' → 5

        starting_date=datetime.strptime(date_str + hour_str, "%Y%m%d%H")

def init_starting_date():
    global starting_date
    image_input_folder="./reasoning/humidity/humidity_at_100m/"

    # Loop through all images
    for filename in sorted(os.listdir(image_input_folder)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        get_starting_date(filename)

def plot_locations_to_map(image_input_folder, image_output_folder, coordinates):
    # Load locations
    global locations_name_px_pos, starting_date

    locations=locations_name_px_pos
    # Unpack coordinates
    
    # Make sure output folder exists
    os.makedirs(image_output_folder, exist_ok=True)
    
    TARGET_WIDTH = 668
    TARGET_HEIGHT = 585  
    # Loop through all images
    for filename in sorted(os.listdir(image_input_folder)):
        print("plot_locations_to_map FOLDER", image_input_folder)

        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        get_starting_date(filename)

        input_path = os.path.join(image_input_folder, filename)
        output_path = os.path.join(image_output_folder, filename)

        # -------------------------------------------------------------
        # 1. Load & resize WITH PIL (pixel-exact)
        # -------------------------------------------------------------
        img = Image.open(input_path)
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        img_np = np.array(img)

        # -------------------------------------------------------------
        # 2. Create exact-size figure in inches for DPI=100
        # -------------------------------------------------------------
        fig_width_in  = TARGET_WIDTH / 100
        fig_height_in = TARGET_HEIGHT / 100

        fig, ax = plt.subplots(
            figsize=(fig_width_in, fig_height_in),
            dpi=100
        )

        # Remove all margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # -------------------------------------------------------------
        # 3. Plot background image
        # -------------------------------------------------------------
        ax.imshow(img_np, origin="upper")

        # -------------------------------------------------------------
        # 4. Plot each location
        # -------------------------------------------------------------
        for location, (px, py) in locations.items():
            ax.plot(px, py, marker=".", color="red", markersize=10)
            ax.text(px + 5, py - 5, location, color="red", fontsize=8)

        # Set limits to prevent shrinking
        ax.set_xlim(0, TARGET_WIDTH)
        ax.set_ylim(TARGET_HEIGHT, 0)

        # Remove axes completely
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        # -------------------------------------------------------------
        # 5. Save image EXACTLY sized
        # -------------------------------------------------------------
        plt.savefig(
            output_path,
            dpi=100,
            bbox_inches=None,   # <-- IMPORTANT (avoid shrinking)
            pad_inches=0        # <-- IMPORTANT
        )
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
                    print(f"Location {loc_name} outside frame {frame_idx} bounds (px={px_i}, py={py_i})")

            frame_cloud_map[frame_idx] = dict(clouds_to_locations)
        except:
            frame_cloud_map[frame_idx]={}

  
    
    return frame_cloud_map

def frame_index_to_timestamp(index, starting_date, frame_time_interval=1):
    """
    Calculate the timestamp of a frame and return (yyyy, mm, dd, h).
    """
    if index < 0:
        raise ValueError("Frame index must be >= 0")
    
    # Convert integer hours to timedelta if necessary
    if isinstance(frame_time_interval, int):
        frame_time_interval = timedelta(hours=frame_time_interval)
    
    timestamp = starting_date + index * frame_time_interval
    return timestamp.year, timestamp.month, timestamp.day, timestamp.hour
    
def generate_cloud_facts_over_cities(base_path):

    load_locations(coordinates,base_path)

    for level, suff in folders_suff.items():
        entry= folder_types[0] + suff
        full_path = os.path.join(base_path, folder_types[0] + suff)

        if os.path.isdir(full_path):
            #later this first stage can be skipped
            plot_locations_to_map(full_path,"reasoning/clouds/"+entry,coordinates)

            #########################################################################
            #compute the map that for each cloud tells me which locations I am covering
            trajectories =  pd.read_csv(full_path+"/trajectories.csv")
            frame_cloud_map = get_clouds_covering_locations(locations_name_px_pos,full_path+"/segment_labels_all.npz",trajectories)
            
            #########################################################################

            # Example: print clouds covering each location in frame 0
            #full_str="starting date: "+str(starting_date)+"\n"
            full_str="% format: cloud_at(location, cloud_id, yyyy, mm, dd, h).\n\n"
            for key in frame_cloud_map.keys():
                for cell_id, locs in frame_cloud_map[key].items():
                    for location in locs:
                        yyyy,mm,dd,h=frame_index_to_timestamp(key,starting_date,1) #1h between each frame
                        cloud_at_string=f"{full_path.rsplit('/', 1)[-1]}"
                        loc_lower = location.lower().replace(" ", "_")
                       
                        full_str+=f"{cloud_at_string}_covers({location},{cell_id},{yyyy},{mm},{dd},{h}).\n"

            path = "reasoning/clouds/" + full_path.rsplit('/', 1)[-1] + "/clouds_covering.txt"
            with open(path, "w") as f:
                f.write(full_str)
            frame_cloud_map={}

    generate_cloud_split_merges()

def generate_cloud_split_merges():
    pass

def rewrite_facts_no_dates(lines):
    """
    Generic transformer:
    - Extracts the timestamp (last four comma-separated fields)
    - Removes them from each predicate
    - Returns (timestamp_fact, stripped_predicates)
    """

    timestamp = None
    stripped_facts = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue

        # Remove trailing ")." safely
        if line.endswith(")."):
            body = line[:-2]
        elif line.endswith(")"):
            body = line[:-1]
        else:
            body = line

        # Split predicate name and arguments
        pred_name, args_str = body.split("(", 1)
        args = args_str.split(",")

        # Last 4 arguments are the timestamp
        if len(args) < 4:
            # malformed line, skip
            continue

        *arg_parts, yyyy, mm, dd, hh = args

        # Create timestamp only once per frame
        if timestamp is None:
            timestamp = f"date_time({yyyy},{mm},{dd},{hh})."

        # Rebuild predicate with remaining args
        new_pred = f"{pred_name}(" + ",".join(arg_parts) + ")."
        stripped_facts.append(new_pred)

    return timestamp, stripped_facts


def merge_into_examples(folder_list_clouds,folder_list_hum, folder_list_temp):
    global starting_date
    folder_split_merge="./image_processing/fvg/output" #ha le subfolder per ogni livello (considera solo le clouds)
    folder_clouds="./reasoning/clouds" #ha le subfolder per ogni livello
    folder_hum="./reasoning/humidity" #ha le subfolder per ogni livello
    folder_temp="./reasoning/temp" #ha le subfolder per ogni livello
    folder_pictograms="./reasoning/pictogram_extraction/extracted"
    
    output_folder="./reasoning/generated_examples"

    num_frames=None
    cloud_covering_data = None
    frame_strings = {}
    for cloud_folder_name,_ in folder_list_clouds:
        # Build the full path to the cloud folder
        full_cloud_folder = os.path.join(folder_clouds, cloud_folder_name)

        if not os.path.isdir(full_cloud_folder):
            print(f"Skipping missing folder: {full_cloud_folder}")
            continue

        if(num_frames is None): #once
            png_files = [f for f in os.listdir(full_cloud_folder) if f.lower().endswith('.png')]
            num_frames = len(png_files)
            cloud_covering_data = {frame_idx: [] for frame_idx in range(num_frames)}

            for frame_idx in range(num_frames):
                y, m, d, h = frame_index_to_timestamp(frame_idx,starting_date)
                # Format like in your file: yyyy,mm,dd,hh
                ts_str = f"{y},{m},{d},{h})"
                frame_strings[frame_idx] = ts_str
  
        #open file "clouds_covering.txt" to get which clouds cover which locations
        cloud_covering_file = os.path.join(full_cloud_folder, "clouds_covering.txt")
        
                
        # Precompute frame timestamp strings
    
        # Scan line by line
        with open(cloud_covering_file, 'r') as f:
            for line in f:
                line = line.strip()
                for frame_idx, ts_str in frame_strings.items():
                    if ts_str in line:
                        cloud_covering_data[frame_idx].append(line)
                        break  # if a line can only belong to one frame

    hum_front_data = {frame_idx: [] for frame_idx in range(num_frames)}
    hum_data = {frame_idx: [] for frame_idx in range(num_frames)}
    
    for hum_folder_name,_ in folder_list_hum:
        full_hum_folder = os.path.join(folder_hum, hum_folder_name)

        if not os.path.isdir(full_hum_folder):
            print(f"Skipping missing folder: {full_hum_folder}")
            continue
        hum_front_file = os.path.join(full_hum_folder, "humidity_fronts.txt")
        hum_file=os.path.join(full_hum_folder, "humidity.txt")

        with open(hum_front_file, 'r') as f:
            for line in f:
                line = line.strip()
                for frame_idx, ts_str in frame_strings.items():
                    if ts_str in line:
                        hum_front_data[frame_idx].append(line)
                        break  # if a line can only belong to one frame

        with open(hum_file, 'r') as f:
            for line in f:
                line = line.strip()
                for frame_idx, ts_str in frame_strings.items():
                    if ts_str in line:
                        hum_data[frame_idx].append(line)                        
                        break  # if a line can only belong to one frame
                    
    temp_data = {frame_idx: [] for frame_idx in range(num_frames)}
    
    for temp_folder_name,_ in folder_list_temp:
        full_temp_folder = os.path.join(folder_temp, temp_folder_name)

        if not os.path.isdir(full_temp_folder):
            print(f"Skipping missing folder: {full_temp_folder}")
            continue
        temp_front_file = os.path.join(full_temp_folder, "temp_fronts.txt")
        temp_file = os.path.join(full_temp_folder, "temp.txt")

        with open(temp_file, 'r') as f:
            for line in f:
                line = line.strip()
                for frame_idx, ts_str in frame_strings.items():
                    if ts_str in line:
                        hum_front_data[frame_idx].append(line)
                        break  # if a line can only belong to one frame
        with open(temp_front_file, 'r') as f:
            for line in f:
                line = line.strip()
                for frame_idx, ts_str in frame_strings.items():
                    if ts_str in line:
                        temp_data[frame_idx].append(line)
                        break  # if a line can only belong to one frame
    
    for frame_idx in range(num_frames):
        output_path = os.path.join(output_folder, f"example_frame_{frame_idx:03d}.txt")
        os.makedirs(output_folder, exist_ok=True)

        cloud_timestamp, cloud_stripped = rewrite_facts_no_dates(cloud_covering_data[frame_idx])

        # Transform the humidity front data
        hum_front_timestamp, hum_front_stripped = rewrite_facts_no_dates(hum_front_data[frame_idx])

        # Transform the temperature data
        temp_front_timestamp, temp_front_stripped = rewrite_facts_no_dates(temp_data[frame_idx])

        hum_timestamp, hum_stripped = rewrite_facts_no_dates(hum_data[frame_idx])
        # Determine which timestamp to use (they MUST be the same)
        # If some datasets are empty, pick first non-empty
        timestamp = cloud_timestamp or hum_timestamp or temp_front_timestamp or hum_front_timestamp
        
            #get the date as yyyy_mm_dd
        y, m, d, h = frame_index_to_timestamp(frame_idx, starting_date)
        date_str = f"{y}_{m:02d}_{d:02d}"

        positive_facts="% Example generated data for frame {}\n\n".format(frame_idx)
        positive_facts+="#pos(e1,{ \n\n"
        #open the pictogram file for that date
        pictogram_file=os.path.join(folder_pictograms,f"pitt_{date_str}_locations.txt")
        
        negative_facts=""
        if os.path.isfile(pictogram_file):
            with open(pictogram_file, 'r') as f_picto:
                pictogram_lines = f_picto.readlines()
                positive_facts+="".join(pictogram_lines)+"\n"
                for line in pictogram_lines:
                    line=line.strip()
                    negative_facts+=compute_negative_facts(line)
        positive_facts+="}, \n"

        excluded_facts="{\n"

        excluded_facts+=negative_facts



        excluded_facts+="},\n"

        context_facts="{\n"

        context_facts+=timestamp + "\n\n"
        context_facts+="% Cloud coverage data:\n"
        context_facts+="% Cloud_covers(location,cloud_id)\n"

        for fact in cloud_stripped:
            context_facts+=fact + "\n"
        context_facts+="\n"

        context_facts+="% Humidity front data:\n"
        context_facts+="% humidty_front(location_1,location_2): between the two locations there's a sharp change \n"

        for fact in hum_front_stripped:
            context_facts+=fact + "\n"
        context_facts+="\n"

        context_facts+="% hum(location_1,humidity_percentage): the percentage is discretized in 0, 20, 40 ... \n"
        for fact in hum_stripped:
            context_facts+=fact + "\n"
        context_facts+="\n"


        context_facts+="% Temperature data (in kelvin):\n"
        for fact in temp_front_stripped:
            context_facts+=fact + "\n"
        context_facts+="\n"
        context_facts+="}). \n"

        background="""
%general bg rules

# RAINS
rains_at(X) :- forecasted_rain(X, Y), Y > 0.
:- rains_at(X), forecasted_rain(X, 0).   % constraint for “only if”

# CLOUD COVER (if)
sunny_at(X) :- forecasted_sky(X, "sunny").
sunny_at(X) :- forecasted_sky(X, "mostly_clear").
partially_sunny_at(X) :- forecasted_sky(X, "partly_cloudy").
covered_at(X) :- forecasted_sky(X, "mostly_cloudy").
covered_at(X) :- forecasted_sky(X, "cloudy").

#other implication verse
:- sunny_at(X), not forecasted_sky(X, "sunny"), not forecasted_sky(X, "mostly_clear").
:- partially_sunny_at(X), not forecasted_sky(X, "partly_cloudy").
:- covered_at(X), not forecasted_sky(X, "mostly_cloudy"), not forecasted_sky(X, "cloudy").

#only one is true
:- sunny_at(X), partially_sunny_at(X).
:- sunny_at(X), covered_at(X).
:- partially_sunny_at(X), covered_at(X).
                    """
        with open(output_path, 'w') as f_out:
            f_out.write(positive_facts)
            f_out.write(excluded_facts)
            f_out.write(context_facts)
            f_out.write(background)

        print(f"Wrote example data to {output_path}")

def compute_negative_facts(line):

    
    match = fact_pattern.search(line)
    if not match:
        return ""  # not a matching forecast fact

    pred = match.group("pred")
    city = match.group("city")
    raw_value = match.group("value").strip()
    if(line.startswith('%')):   
        return "" #skip comments
    # Case 1: forecasted_rain(..., number)
    if pred == "forecasted_rain":
        value = int(raw_value)  # value is numeric
        if(value != 0):
            return "sunny_at("+city+"). \n"
        else:
            return "rains_at("+city+"). \n"
    # Case 2: forecasted_sky(..., "string")
    else:
        # Remove surrounding quotes if present
        if raw_value.startswith('"') and raw_value.endswith('"'):
            value = raw_value[1:-1]
        else:
            value = raw_value
        
        if value == "sunny" or value == "mostly_clear":
            return "partially_sunny_at("+city+"). \n"+"covered_at("+city+"). \n"
        elif value == "partly_cloudy":
            return "sunny_at("+city+"). \n"+"covered_at("+city+"). \n"
        elif value == "mostly_cloudy" or value == "cloudy":
            return "sunny_at("+city+"). \n"+"partially_sunny_at("+city+"). \n"
    return ""




def color_to_humidity(rgb_color, legend_colors, legend_values):
    """
    Given an (R,G,B) color, find the closest color in the legend
    and return its corresponding humidity value.
    """
    color = np.array(rgb_color)
    dists = np.linalg.norm(legend_colors - color, axis=1)
    idx = np.argmin(dists)
    return float(legend_values[idx])

def generate_humidity_facts_over_cities(base_path):

    global starting_date
    load_locations(coordinates,base_path)

    for level, suff in folders_suff.items():

        entry = folder_types[1] + suff 
        full_path = os.path.join(base_path, entry)
        
        print("starting date humidity: ", entry)
        #from height to hpa
        match = re.search(r'(\d+(?:[_\.]\d+)?(?:m|km))', entry)
        hpa=str(match.group(1))

        legend_colors, legend_values = load_legend_mapping(f"./raw_data/extracted_fvg_cleaned/legend_at_{hpa}_humidity.png")

        if os.path.isdir(full_path):

            #later this first stage can be skipped
            print("plotting locations to map")
            plot_locations_to_map(full_path,"reasoning/humidity/"+entry,coordinates)

            
            hum_values=get_humidity_over_locations_color(locations_name_px_pos,full_path,legend_colors, legend_values)

            path="reasoning/humidity/" + full_path.rsplit('/', 1)[-1] + "/humidity.txt"
            with open(path, "w") as f:
                f.write("% format humidity_percentage_at(location, humidity_percentage, yyyy, mm, dd, h).\n\n")
                for frame, values in hum_values.items():
                    yyyy,mm,dd,h=frame_index_to_timestamp(frame, starting_date, 1)
                    for location_name, _ in locations_name_px_pos.items():
                        loc_lower = location_name.lower().replace(" ", "_")
                        #witout appproximation use: int(values[location_name])
                        approximated_hum=int(round(values[location_name]/20)*20) #approx to the nearest 10
                        f.write(f"humidity_percentage{suff}({loc_lower}, {approximated_hum}, {yyyy},{mm},{dd},{h}).\n")
                    
def get_humidity_over_locations_color(locations, frames_path, legend_colors, legend_values, radius=5):
    """
    Given city positions and a folder of humidity frames,
    computes average humidity percentage around each location.
    """
    import os
    frame_humidity_map = {}
    frames = sorted([f for f in os.listdir(frames_path) if f.lower().endswith((".png",".jpg",".tif"))])

    for frame_idx, frame_file in enumerate(frames):
        img = np.array(Image.open(os.path.join(frames_path, frame_file)).convert("RGB"))
        h, w, _ = img.shape
        loc_to_hum = {}

        for name, (px, py) in locations.items():
            px_i, py_i = int(round(px)), int(round(py))
            if 0 <= px_i < w and 0 <= py_i < h:
                x_min, x_max = max(0, px_i-radius), min(w, px_i+radius+1)
                y_min, y_max = max(0, py_i-radius), min(h, py_i+radius+1)
                region = img[y_min:y_max, x_min:x_max]

                mean_color = region.reshape(-1,3).mean(axis=0)
                hum = color_to_humidity(mean_color, legend_colors, legend_values)
                loc_to_hum[name] = hum
            else:
                print(f"{name} outside frame {frame_idx}")
        frame_humidity_map[frame_idx] = loc_to_hum
    return frame_humidity_map

def load_legend_mapping(legend_path, n_samples=101,min_value=0, max_value=100):
    """
    Reads a horizontal colorbar legend (e.g. humidity 0–100%),
    returns (legend_colors, legend_values).
    """
    legend_img = np.array(Image.open(legend_path).convert("RGB"))
    h, w, _ = legend_img.shape

    # Average vertically, since the color bar may have thickness
    avg_colors = legend_img.mean(axis=0)  # shape (w,3)

    # Sample N equally spaced positions along the width
    idx = np.linspace(0, w - 1, n_samples).astype(int)
    legend_colors = avg_colors[idx]
    legend_values = np.linspace(min_value, max_value, n_samples)

    return legend_colors, legend_values

def generate_temp_facts_over_cities(base_path):

    global starting_date
    load_locations(coordinates,base_path)

    for level, suff in folders_suff.items():

        entry = folder_types[2] + suff 
        full_path = os.path.join(base_path, entry)
        
        #from height to hpa
        match = re.search(r'(\d+(?:[_\.]\d+)?(?:m|km))', entry)
        hpa=str(match.group(1))


        #open the file with min and max temp
        with open(f"./raw_data/extracted_fvg_cleaned/temp{folders_suff[level]}/legend{folders_suff[level]}_temp.txt", 'r') as ftxt:
            content = ftxt.read()

        match = re.search(r"Temperature range at \d+\s* hPa:\s*([\d.]+)\s*K\s*to\s*([\d.]+)\s*K", content)
        min_temp=None
        max_temp=None
        if match:
            min_temp = int(float(match.group(1)))
            max_temp = int(float(match.group(2)))
            print("temperatures:    ",  min_temp, max_temp)
        else:
            raise ValueError("No temperature range found in legend file.")  

        sample_points=max_temp - min_temp +1 #1 step each degree
        legend_colors, legend_values = load_legend_mapping(f"./raw_data/extracted_fvg_cleaned/legend_at_{hpa}_temp.png", n_samples=sample_points, min_value=min_temp, max_value=max_temp)

        if os.path.isdir(full_path):

            #later this first stage can be skipped
            plot_locations_to_map(full_path,"reasoning/temp/"+entry,coordinates)
            print("getting humidity over locations color FULL PATH", full_path)
            
            hum_values=get_humidity_over_locations_color(locations_name_px_pos,full_path,legend_colors, legend_values)
            path="reasoning/temp/" + full_path.rsplit('/', 1)[-1] + "/temp.txt"
            with open(path, "w") as f:
                f.write("% format temperature_at(location, temperature, yyyy, mm, dd, h).\n\n")
                for frame, values in hum_values.items():
                    yyyy,mm,dd,h=frame_index_to_timestamp(frame, starting_date, 1)
                    for location_name, _ in locations_name_px_pos.items():
                        loc_lower = location_name.lower().replace(" ", "_")
                        #witout appproximation use: int(values[location_name])
                        approximated_temp=round(values[location_name]) #approx (should already be int)
                        f.write(f"temperature_at{suff}({loc_lower}, {approximated_temp}, {yyyy},{mm},{dd},{h}).\n")
    return starting_date