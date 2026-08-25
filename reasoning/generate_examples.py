import os
import re
import shutil
from datetime import datetime

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# longmin longmax latmin latmax of FVG and Italy, longmin longmax, latmin latmax
coordinates = [11, 15, 44.5, 48]
coordinates_italy = [10, 16, 42, 48]
folder_types = ["cloud", "humidity", "temp"]

folders_suff = {
    1000: "_at_100m",
    925: "_at_750m",
    850: "_at_1_4km",
    700: "_at_3km",
    500: "_at_5_5km",
    300: "_at_9km"
}
locations_name_px_pos = {}
location_names = []

fact_pattern = re.compile(
    r'(?P<pred>forecasted_rain|forecasted_sky)\('
    r'\s*(?P<city>[a-zA-Z_]+)\s*,\s*'
    r'(?P<value>[^)]+)\)'
)

starting_date = None


def feature_to_cell_id(feature_id, trajectories):
    # print("looking for feature ->",feature_id)
    feature_to_cell_map = trajectories.set_index("feature")["cell"].to_dict()
    return feature_to_cell_map.get(feature_id, None)


def geo_to_pixel(lon, lat, lon_min, lon_max, lat_min, lat_max, img_width, img_height):
    x = (lon - lon_min) / (lon_max - lon_min) * img_width
    y = (lat_max - lat) / (lat_max - lat_min) * img_height
    return x, y


def load_locations(coordinates, image_input_folder):
    global locations_name_px_pos
    locations_data = pd.read_csv("reasoning/locations.csv")  # same folder or specify full path
    # print(f"Loaded {len(locations_data)} locations.")

    lon_min, lon_max = coordinates[0], coordinates[1]
    lat_min, lat_max = coordinates[2], coordinates[3]

    # read a file just to get the image shape (they are all =)
    first_dir = sorted(os.listdir(image_input_folder))[0]
    first_dir = image_input_folder + first_dir
    first_file = sorted([f for f in os.listdir(first_dir) if
                         os.path.isfile(os.path.join(first_dir, f)) and f.lower().endswith(
                             (".png", ".jpg", ".jpeg", ".tif", ".tiff"))])[0]

    input_path = os.path.join(first_dir, first_file)

    # Load image
    img = mpimg.imread(input_path)
    height, width = img.shape[:2]

    for _, row in locations_data.iterrows():
        px, py = geo_to_pixel(
            row["Long"], row["Lat"],
            lon_min, lon_max,
            lat_min, lat_max,
            width, height
        )
        locations_name_px_pos[row["Location"]] = (px, py)
    return locations_name_px_pos


def load_location_names():
    global location_names
    locations_data = pd.read_csv("reasoning/locations.csv")  # same folder or specify full path
    print(f"Loaded {len(locations_data)} locations.")

    for _, row in locations_data.iterrows():
        location_names.append((row["Location"]).lower())


def plot_vectors_and_locations(df, locations_name_px_pos):
    plt.figure(figsize=(10, 10))

    # --- Plot wind vectors ---
    plt.scatter(
        df["pixel_x_scaled"],
        df["pixel_y_scaled"],
        s=10,
        alpha=0.6,
        label="Wind vectors"
    )

    # Label each vector with its ID (can be noisy!)
    for _, row in df.iterrows():
        plt.text(
            row["pixel_x_scaled"],
            row["pixel_y_scaled"],
            str(int(row["vector_id"])),
            fontsize=6,
            alpha=0.6
        )

    # --- Plot city locations ---
    for city, (cx, cy) in locations_name_px_pos.items():
        plt.scatter(cx, cy, marker="x", s=100)
        plt.text(cx + 3, cy + 3, city, fontsize=9, weight="bold")

    # --- Formatting ---
    plt.title("City locations and wind vectors (pixel space)")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.gca().invert_yaxis()  # IMPORTANT for image coordinates
    plt.legend()
    plt.grid(True)

    plt.savefig("wind_vectors_debug.png", dpi=200, bbox_inches="tight")
    plt.close()


def get_starting_date(filename):
    global starting_date
    if (starting_date is None):
        parts = filename.split("_")
        date_str = parts[-2]  # '20191101'
        hour_str = parts[-1][:2]  # '05' → 5

        starting_date = datetime.strptime(date_str + hour_str, "%Y%m%d%H")


def init_starting_date():
    global starting_date
    image_input_folder = "./reasoning/humidity/humidity_at_100m/"

    # Loop through all images
    for filename in sorted(os.listdir(image_input_folder)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        get_starting_date(filename)


def plot_locations_to_map(image_input_folder, image_output_folder, coordinates):
    # Load locations
    global locations_name_px_pos, starting_date

    locations = locations_name_px_pos
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
        fig_width_in = TARGET_WIDTH / 100
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
            bbox_inches=None,  # <-- IMPORTANT (avoid shrinking)
            pad_inches=0  # <-- IMPORTANT
        )
        plt.close(fig)

        print(f"Saved {output_path}")


def get_clouds_covering_locations(locations_name_px_pos, segment_labels_path, trajectories):
    """
    Returns a mapping: frame_index -> { cell_id -> [locations] }.
    """

    data = np.load(segment_labels_path, allow_pickle=True)
    print("loading ", segment_labels_path)

    segment_labels_list = [data[key] for key in data]

    frame_cloud_map = {}

    for frame_idx, seg_labels in enumerate(segment_labels_list):
        # --- Handle non-2D arrays safely ---
        try:  # to deal with empty frames (i.e. frames without blobs)
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
                    if (feature_id > 0):
                        cell_id = feature_to_cell_id(int(feature_id), trajectories)

                        clouds_to_locations[int(cell_id)].append(loc_name)
                else:
                    print(f"Location {loc_name} outside frame {frame_idx} bounds (px={px_i}, py={py_i})")

            frame_cloud_map[frame_idx] = dict(clouds_to_locations)
        except:
            frame_cloud_map[frame_idx] = {}

    return frame_cloud_map


def get_sorted_png_files(folder_path):
    print("Getting PNG files from:", folder_path)
    return sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".png")
    )


def extract_timestamp_from_filename(filename):
    """
    Extract (yyyy, mm, dd, hh) from filenames like:
    cloud_1000_20190401_0300.png
    """
    match = re.search(r'(\d{8})_(\d{4})', filename)
    if not match:
        raise ValueError(f"Cannot extract timestamp from {filename}")

    date_str, hour_str = match.groups()

    yyyy = int(date_str[0:4])
    mm = int(date_str[4:6])
    dd = int(date_str[6:8])
    hh = int(hour_str[0:2])

    return yyyy, mm, dd, hh


def segment_direction(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y1 - y2  # invert Y because image Y grows downward

    if dx == 0 and dy == 0:
        raise ValueError("Points must be different")

    # Angle in degrees: 0° = East, CCW positive
    angle = math.degrees(math.atan2(dy, dx))
    angle = (angle + 360) % 360

    directions = [
        "e",  # 0°
        "ne",  # 45°
        "n",  # 90°
        "nw",  # 135°
        "w",  # 180°
        "sw",  # 225°
        "s",  # 270°
        "se"  # 315°
    ]

    index = int((angle + 22.5) // 45) % 8
    return directions[index]


def generate_cloud_movements(base_path):
    pattern = r"Frame (\d+), cell (\d+) moved from \(x:\s*([\d.]+),\s*y:([\d.]+)\s*\) to \(x:\s*([\d.]+),\s*y:([\d.]+)\s*\)"
    for level, suff in folders_suff.items():
        entry = folder_types[0] + suff
        full_path = os.path.join(base_path, folder_types[0] + suff)
        full_str = "% format: cloud_moves(cell_id, direction, yyyy, mm, dd, h).\n\n"
        with open(full_path + "/movements.txt", "r") as f:
            png_files = get_sorted_png_files(full_path)
            for line in f:
                line = line.strip()
                if not line:
                    continue  # no empty ones 
                match = re.search(pattern, line)
                if match:
                    # Estrazione dei dati
                    frame_num = match.group(1)
                    cell_id = match.group(2)
                    x_start = float(match.group(3))
                    y_start = float(match.group(4))
                    x_end = float(match.group(5))
                    y_end = float(match.group(6))
                    direction = segment_direction(x_start, y_start, x_end, y_end)

                    # Generate facts for this movement
                    filename = png_files[int(frame_num)]
                    yyyy, mm, dd, h = extract_timestamp_from_filename(filename)
                    full_str += f"{full_path.rsplit('/', 1)[-1]}_moves({cell_id},{direction},{yyyy},{mm},{dd},{h}).\n"

        path = "reasoning/clouds/" + full_path.rsplit('/', 1)[-1] + "/moving.txt"
        with open(path, "w") as f:
            f.write(full_str)


def generate_cloud_facts_over_cities(base_path):
    load_locations(coordinates, base_path)

    for level, suff in folders_suff.items():
        entry = folder_types[0] + suff
        full_path = os.path.join(base_path, folder_types[0] + suff)

        if os.path.isdir(full_path):
            # later this first stage can be skipped
            plot_locations_to_map(full_path, "reasoning/clouds/" + entry, coordinates)

            #########################################################################
            # compute the map that for each cloud tells me which locations I am covering
            trajectories = pd.read_csv(full_path + "/trajectories.csv")
            frame_cloud_map = get_clouds_covering_locations(locations_name_px_pos,
                                                            full_path + "/segment_labels_all.npz", trajectories)

            #########################################################################

            # Example: print clouds covering each location in frame 0
            # full_str="starting date: "+str(starting_date)+"\n"
            full_str = "% format: cloud_at(location, cloud_id, yyyy, mm, dd, h).\n\n"
            png_files = get_sorted_png_files(full_path)
            for key in frame_cloud_map.keys():
                for cell_id, locs in frame_cloud_map[key].items():
                    for location in locs:
                        filename = png_files[key]
                        yyyy, mm, dd, h = extract_timestamp_from_filename(filename)
                        cloud_at_string = f"{full_path.rsplit('/', 1)[-1]}"
                        loc_lower = location.lower().replace(" ", "_")

                        full_str += f"{cloud_at_string}_covers({loc_lower},{cell_id},{yyyy},{mm},{dd},{h}).\n"

            path = "reasoning/clouds/" + full_path.rsplit('/', 1)[-1] + "/clouds_covering.txt"
            with open(path, "w") as f:
                f.write(full_str)
            frame_cloud_map = {}

    generate_cloud_split_merges()


def generate_cloud_split_merges():
    pass


import re


def get_fronts(front_data, location_names, hum_or_tmp):
    """
    front_data: list of strings (facts for ONE day)
    location_names: set/list of valid locations
    hum_or_tmp: "hum" or "temp"
    """

    from collections import defaultdict
    import re

    fronts = defaultdict(list)
    result = []

    # Example:
    # hum_front_500_hPa(l1,l2,YYYY,MM,DD,HH).
    front_re = re.compile(
        r"(hum|temp)_front_(\d+)_hPa\(([^,]+),([^,]+),(\d+),(\d+),(\d+),(\d+)\)"
    )

    for line in front_data:
        line = line.strip()
        if not line or line.startswith("%"):
            continue

        if line.endswith("."):
            line = line[:-1]

        m = front_re.match(line)
        if not m:
            continue

        _, height, l1, l2, yyyy, mm, dd, hh = m.groups()

        if l1 not in location_names or l2 not in location_names:
            continue

        try:
            hour = int(hh)
        except ValueError:
            continue

        # Keep only daytime hours
        if hour < 7 or hour > 19:
            continue

        period = "morning" if hour < 12 else "afternoon"

        key = (l1, l2, period, height)
        fronts[key].append(hour)

    # Generate ONE fact if there exist ≥3 consecutive hours
    for (l1, l2, period, height), hours in fronts.items():
        hours = sorted(set(hours))

        has_3_consecutive = any(
            h2 == h1 + 1 and h3 == h2 + 1
            for h1, h2, h3 in zip(hours, hours[1:], hours[2:])
        )

        if has_3_consecutive:
            result.append(
                f"{hum_or_tmp}_{period}{folders_suff[int(height)]}({l1},{l2})."
            )

    return result


def sum_up_morning_afternoon(temp_data, location_names, hum_or_tmp):
    """
    Transformer:
    - Removes timestamp (last four comma-separated fields)
    - Skips hours <7 or >19
    - Adds "m" if 7 <= hour < 12, "a" if 12 <= hour <= 19
    - Collects stripped predicates
    - Computes per-location average temperatures for morning and afternoon
      truncated to 2 digits after the decimal
    """

    stripped_facts = []
    # Store temperatures per location and period
    temps = defaultdict(lambda: {"morning": [], "afternoon": []})

    for line in temp_data:
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

        if "(" not in body:
            continue  # malformed
        pred_name, args_str = body.split("(", 1)
        args = [a.strip() for a in args_str.split(",")]

        # Must have at least 5 arguments for location + temperature + timestamp
        if len(args) < 5:
            continue

        *arg_parts, yyyy, mm, dd, hh = args

        # Assume first argument is location, second argument is temperature
        loc = arg_parts[0]
        try:
            temp = float(arg_parts[1])
            hour = int(hh)
        except ValueError:
            continue  # skip malformed data

        # Skip hours outside 7..19
        if hour < 7 or hour > 19:
            continue

        # Determine period
        period = "morning" if hour < 12 else "afternoon"

        # Rebuild predicate with period as last argument
        time_label = f"\"{period}\""
        new_pred = f"{pred_name}(" + ",".join(arg_parts + [time_label]) + ")."
        stripped_facts.append(new_pred)

        # Add temperature to period collection if location is relevant
        if loc in location_names:
            temps[loc][period].append(temp)

    # Compute average temperature per location and period
    average_facts = []
    avg_truncated_morning = 0
    avg_truncated_afternoon = 0
    for loc, periods in temps.items():
        temps_list_morning = periods["morning"]
        if temps_list_morning:  # avoid division by zero
            avg = sum(temps_list_morning) / len(temps_list_morning)
            avg_truncated_morning = f"{avg:.2f}"
            avg_fact = f"% {hum_or_tmp}_at_morning({loc},{avg_truncated_morning})."
            average_facts.append(avg_fact)
        temps_list_afternoon = periods["afternoon"]
        if temps_list_afternoon:  # avoid division by zero
            avg = sum(temps_list_afternoon) / len(temps_list_afternoon)
            avg_truncated_afternoon = f"{avg:.2f}"
            avg_fact = f"% {hum_or_tmp}_at_afternoon({loc},{avg_truncated_afternoon})."
            average_facts.append(avg_fact)
        static_fact = ""
        if float(avg_truncated_morning) <= float(avg_truncated_afternoon):
            static_fact = f"{hum_or_tmp}_increased_at_afternoon({loc})."
        else:
            static_fact = f"{hum_or_tmp}_decreased_at_afternoon({loc})."
        average_facts.append(static_fact)
    return average_facts


import math

DIR_TO_ANGLE = {
    "E": 0,
    "NE": 45,
    "N": 90,
    "NW": 135,
    "W": 180,
    "SW": 225,
    "S": 270,
    "SE": 315,
}

ANGLE_TO_DIR = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def angle_to_compass(angle_deg):
    angle = angle_deg % 360
    idx = int((angle + 22.5) // 45) % 8
    return ANGLE_TO_DIR[idx]


from collections import defaultdict


def sum_up_morning_afternoon_winds(wind_data, location_names):
    """
    - Skips hours <7 or >19
    - Morning: 7–11, Afternoon: 12–19
    - Averages wind vectors (u,v), NOT magnitudes
    - Produces one wind fact per location per period
    """
    print("Summing up winds for morning and afternoon...")
    print(wind_data)
    # Store u,v components per location and period
    winds = defaultdict(lambda: {
        "morning": {"u": [], "v": []},
        "afternoon": {"u": [], "v": []}
    })

    for line in wind_data:
        line = line.strip()
        if not line or line.startswith('%'):
            continue

        # Strip ")."
        body = line[:-2] if line.endswith(").") else line.rstrip(")")
        if "(" not in body:
            continue

        pred_name, args_str = body.split("(", 1)
        args = [a.strip() for a in args_str.split(",")]

        # Expect: loc, dir, speed, yyyy, mm, dd, hh
        if len(args) != 7:
            continue

        loc, direction, speed, yyyy, mm, dd, hh = args

        if loc not in location_names:
            continue

        try:
            speed = float(speed)
            hour = int(hh)
        except ValueError:
            continue

        if hour < 7 or hour > 19:
            continue

        period = "morning" if hour < 12 else "afternoon"

        if direction not in DIR_TO_ANGLE:
            continue

        # Convert to vector
        angle_rad = math.radians(DIR_TO_ANGLE[direction])
        u = speed * math.cos(angle_rad)
        v = speed * math.sin(angle_rad)

        winds[loc][period]["u"].append(u)
        winds[loc][period]["v"].append(v)

    # --- Build averaged facts ---
    final_facts = []
    max_speed_overall = 0
    for loc, periods in winds.items():
        for period in ["morning", "afternoon"]:
            u_list = periods[period]["u"]
            v_list = periods[period]["v"]

            if not u_list:
                continue

            u_avg = sum(u_list) / len(u_list)
            v_avg = sum(v_list) / len(v_list)

            speed_avg = math.sqrt(u_avg ** 2 + v_avg ** 2)

            if speed_avg < 1e-6:  # calm wind
                direction_avg = "calm"
            else:
                angle_avg = math.degrees(math.atan2(v_avg, u_avg))
                direction_avg = angle_to_compass(angle_avg)

            speed_avg = int(speed_avg)  # o :.3f
            max_speed_overall = max(max_speed_overall, speed_avg)
            final_facts.append(
                f'wind_blowing_{period}({loc},"{direction_avg}",{speed_avg}).'
            )
    print("Max average wind speed found:", max_speed_overall)
    return final_facts


def rewrite_facts_no_dates(lines):
    """
    Generic transformer:
    - Extracts the timestamp (last four comma-separated fields)
    - Removes them from each predicate
    - Returns (timestamp_fact, stripped_predicates)
    """

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

        # Rebuild predicate with remaining args
        new_pred = f"{pred_name}(" + (",".join(arg_parts)) + f", {str(hh)})."
        stripped_facts.append(new_pred)

    return stripped_facts


def get_all_dates():
    pictogram_folder = "./reasoning/pictogram_extraction/pictograms/sky"
    # for eacj file in the folder get the date
    date_list = []
    for filename in sorted(os.listdir(pictogram_folder)):
        if not filename.lower().endswith((".png")):
            continue
        parts = filename.split("_")
        # file is like 2019_11_01.png
        # remove .png from last part
        parts[-1] = parts[-1][:-4]  # '20191101'
        date_list.append((int(parts[0]), int(parts[1]), int(parts[2])))

    return date_list


def calculate_winds(date, file_str, suffix, hh):
    global locations_name_px_pos
    load_locations(coordinates, "./reasoning/clouds/")

    df = pd.read_csv(file_str)  # adjust filename
    # --- Rescale CSV pixel coordinates ---
    df["pixel_x_scaled"] = df["pixel_x"] / 4.0
    df["pixel_y_scaled"] = df["pixel_y"] / 4.0
    # plot_vectors_and_locations(df, locations_name_px_pos)
    # --- City pixel coordinates ---

    # --- Find nearest wind vector for each city ---
    results = {}

    for city, (cx, cy) in locations_name_px_pos.items():
        distances = np.sqrt(
            (df["pixel_x_scaled"] - cx) ** 2 +
            (df["pixel_y_scaled"] - cy) ** 2
        )

        idx = distances.idxmin()
        results[city] = df.loc[idx]

    # --- Display results ---
    facts = []
    date_y, date_m, date_d = date
    for city, row in results.items():
        direction = angle_to_compass(row.alpha_deg)
        # get the magnitude if needed: row.magnitude
        magnitude = row.magnitude if "magnitude" in df.columns else None
        fact = f"wind_blowing{suffix}({city.lower()}, {direction}, {magnitude}, {date_y},{date_m},{date_d},{hh})."
        facts.append(fact)

    return facts


def angle_to_compass(alpha_deg):
    """
    Convert angle in degrees to one of:
    N, NE, E, SE, S, SW, W, NW
    Assumes:
      - 0° = East
      - 90° = North
      - CCW positive
    """
    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]

    # Normalize angle to [0, 360)
    alpha = alpha_deg % 360

    # Each sector is 45°
    idx = int((alpha + 22.5) // 45) % 8
    return directions[idx]


def merge_into_examples(folder_list_clouds, folder_list_hum, folder_list_temp, folder_list_wind, folders_suff):
    global starting_date
    global location_names

    print(folder_list_clouds)
    print(folder_list_wind)

    folder_split_merge = "./image_processing/fvg/output"  # ha le subfolder per ogni livello (considera solo le clouds)
    folder_clouds = "./reasoning/clouds"  # ha le subfolder per ogni livello
    folder_hum = "./reasoning/humidity"  # ha le subfolder per ogni livello
    folder_temp = "./reasoning/temp"  # ha le subfolder per ogni livello
    folder_pictograms = "./reasoning/pictogram_extraction/extracted"
    folder_winds = "./raw_data/extracted_fvg_cleaned"  # ha le subfolder per ogni livello
    output_folder = "./reasoning/generated_examples"

    date_list = get_all_dates()

    num_frames = None
    cloud_covering_data = None
    cloud_moving_data = None
    frame_strings = {}

    for cloud_folder_name, _ in folder_list_clouds:
        # Build the full path to the cloud folder
        full_cloud_folder = os.path.join(folder_clouds, cloud_folder_name)

        if not os.path.isdir(full_cloud_folder):
            print(f"Skipping missing folder: {full_cloud_folder}")
            continue

        if (num_frames is None):  # once
            png_files = [f for f in os.listdir(full_cloud_folder) if f.lower().endswith('.png')]
            num_frames = len(png_files)
            cloud_covering_data = {date: [] for date in date_list}
            cloud_moving_data = {date: [] for date in date_list}

            for (y, m, d) in date_list:
                # Format like in your file: yyyy,mm,dd
                ts_str = f"{int(y)},{int(m)},{int(d)}"
                frame_strings[(int(y), int(m), int(d))] = ts_str

        # open file "clouds_covering.txt" to get which clouds cover which locations
        cloud_covering_file = os.path.join(full_cloud_folder, "clouds_covering.txt")
        cloud_moving_file = os.path.join(full_cloud_folder, "moving.txt")
        # print("Processing cloud folder:", cloud_covering_file)

        # Precompute frame timestamp strings

        print("date_days ", date_list)
        # Scan line by line
        with open(cloud_covering_file, 'r') as f:
            for line in f:
                line = line.strip()
                for date_day, ts_str in frame_strings.items():
                    if f"{ts_str}," in line:
                        cloud_covering_data[date_day].append(line)
                        break  # if a line can only belong to one frame

        with open(cloud_moving_file, 'r') as f:
            for line in f:
                line = line.strip()
                for date_day, ts_str in frame_strings.items():
                    if f"{ts_str}," in line:
                        cloud_moving_data[date_day].append(line)
                        break  # if a line can only belong to one frame

    print("Finished processing cloud data.")
    # print(cloud_covering_data)

    hum_front_data = {date: [] for date in date_list}
    hum_data = {date: [] for date in date_list}

    for hum_folder_name, _ in folder_list_hum:
        full_hum_folder = os.path.join(folder_hum, hum_folder_name)

        if not os.path.isdir(full_hum_folder):
            print(f"Skipping missing folder: {full_hum_folder}")
            continue
        hum_front_file = os.path.join(full_hum_folder, "humidity_fronts.txt")
        hum_file = os.path.join(full_hum_folder, "humidity.txt")

        with open(hum_front_file, 'r') as f:
            for line in f:
                line = line.strip()
                for date_day, ts_str in frame_strings.items():
                    if ts_str in line:
                        hum_front_data[date_day].append(line)
                        break  # if a line can only belong to one frame
        with open(hum_file, 'r') as f:
            for line in f:
                line = line.strip()
                for date_day, ts_str in frame_strings.items():
                    if f"{ts_str}," in line:
                        hum_data[date_day].append(line)
                        break  # if a line can only belong to one frame

    temp_data = {date: [] for date in date_list}

    temp_front_data = {date: [] for date in date_list}
    for temp_folder_name, _ in folder_list_temp:
        full_temp_folder = os.path.join(folder_temp, temp_folder_name)

        if not os.path.isdir(full_temp_folder):
            print(f"Skipping missing folder: {full_temp_folder}")
            continue

        temp_front_file = os.path.join(full_temp_folder, "temp_fronts.txt")
        temp_file = os.path.join(full_temp_folder, "temp.txt")

        with open(temp_file, 'r') as f:
            for line in f:
                line = line.strip()
                for date_day, ts_str in frame_strings.items():
                    if f"{ts_str}," in line:
                        temp_data[date_day].append(line)
                        break  # if a line can only belong to one frame

        with open(temp_front_file, 'r') as f:
            for line in f:
                line = line.strip()
                for date_day, ts_str in frame_strings.items():
                    if ts_str in line:
                        temp_front_data[date_day].append(line)
                        break  # if a line can only belong to one frame

    wind_data = {date: [] for date in date_list}

    for win_folder_name, _ in folder_list_wind:
        full_wind_folder = os.path.join(folder_winds, win_folder_name)

        if not os.path.isdir(full_wind_folder):
            print(f"Skipping missing folder: {full_wind_folder}")
            continue

        for date in date_list:
            y, m, d = date
            hPa = next((k for k, v in folders_suff.items() if v in win_folder_name), None)
            suffix = folders_suff[hPa]
            for hour in range(1, 24):
                wind_file = os.path.join(full_wind_folder, f"wind_{hPa}_{y:04d}{m:02d}{d:02d}_{hour:02d}00.csv")
                wind_data[date].extend(calculate_winds(date, wind_file, suffix, hour))
                # print("Processing wind file:", wind_file)

    i = 0
    load_location_names()
    for city in location_names:

        for date in date_list:

            y, m, d = date
            output_path = os.path.join(output_folder, f"example_day_{y}_{m}_{d}_{city}.las")
            os.makedirs(output_folder, exist_ok=True)
            print("Generating example for date:", date)
            cloud_stripped = rewrite_facts_no_dates(cloud_covering_data[date])
            temp_morning_afternoon = sum_up_morning_afternoon(temp_data[date], location_names, "temperature")
            hum_morning_afternoon = sum_up_morning_afternoon(hum_data[date], location_names, "humidity")
            temp_fronts = get_fronts(temp_front_data[date], location_names, "temp_front")
            hum_fronts = get_fronts(hum_front_data[date], location_names, "hum_front")
            wind_facts = wind_data[date]
            winds_morning_afternoon = sum_up_morning_afternoon_winds(wind_data[date], location_names)
            # wind_facts_morning=sum_up_morning_afternoon(wind_facts[date], location_names,"wind_blowing")
            # wind_facts_morning=sum_up_morning_afternoon(wind_facts[date], location_names,"wind_blowing")

            # cloud_moving_timestamp, cloud_moving_stripped = rewrite_facts_no_dates(cloud_moving_data[date])

            # Transform the humidity front data
            # um_front_timestamp, hum_front_stripped = rewrite_facts_no_dates(hum_front_data[date])

            # Transform the temperature data
            # temp_front_timestamp, temp_front_stripped = rewrite_facts_no_dates(temp_data[date])

            # hum_timestamp, hum_stripped = rewrite_facts_no_dates(hum_data[date])

            # Determine which timestamp to use (they MUST be the same)
            # If some datasets are empty, pick first non-empty
            timestamp = "date({}, {}, {}).\n".format(y, m, d)

            m_int = int(m)  # just to be safe

            if m_int == 12 or m_int <= 2:
                season = "winter"  # winter
            elif 3 <= m_int <= 5:
                season = "spring"  # spring
            elif 6 <= m_int <= 8:
                season = "summer"  # summer
            elif 9 <= m_int <= 11:
                season = "autumn"  # autumn

            # get the date as yyyy_mm_dd
            date_str = f"{y}_{m}_{d}"

            positive_facts = "% Example generated data for day {}\n\n".format(date)
            positive_facts += "#pos(e" + str(i) + "@1000,{ \n\n"
            # open the pictogram file for that date
            pictogram_file = os.path.join(folder_pictograms, f"{date_str}_locations.txt")

            negative_facts = []

            if os.path.isfile(pictogram_file):
                with open(pictogram_file, 'r') as f_picto:
                    pictogram_lines = f_picto.readlines()
                    modified_lines = []

                    for line in pictogram_lines:
                        raw = line.rstrip("\n")  # keep original line
                        stripped = raw.strip()  # remove ALL surrounding whitespace
                        if (city in stripped):
                            # Skip comments
                            if stripped.startswith("%") or stripped == "":
                                modified_lines.append(raw + "\n")
                                continue

                            # Match predicate(args) optionally followed by comma
                            m = re.match(r'^(\w+)\(([^)]*)\)(,?)$', stripped)
                            if not m:
                                # Structural line → keep unchanged
                                modified_lines.append(raw + "\n")
                                continue

                            pred, args, comma = m.groups()
                            new_line = f"{pred}({args}, {season}){comma}"
                            modified_lines.append(new_line + "\n")

                    positive_facts += "".join(modified_lines).rstrip(", \n")

                    for line in pictogram_lines:
                        if (city in line):
                            line = line.strip()
                            negative_facts.append(compute_negative_facts(line, season))
            else:
                print(f"Pictogram file not found: {pictogram_file}")
            # print(f"pictogram file {pictogram_file} positive facts for day:{date}: {positive_facts}")
            positive_facts += "},\n"
            negative_facts = "".join(negative_facts).rstrip(", \n")

            excluded_facts = "{\n"
            # print("negative facts:", negative_facts)
            excluded_facts += negative_facts

            excluded_facts += "\n},\n"

            context_facts = "{\n"
            context_facts += f"location_considered({city}). \n"
            context_facts += "%to drive the season (winter, spring, summer, autumn)\n" + timestamp + "\n"
            context_facts += "% Cloud coverage data:\n"
            context_facts += "% Cloud_covers(location,cloud_id,hh)\n"

            for fact in cloud_stripped:
                context_facts += fact + "\n"
            context_facts += "\n%summing up temperature and humidity facts \n\n"
            for fact in temp_morning_afternoon:
                context_facts += fact + "\n"

            for fact in hum_morning_afternoon:
                context_facts += fact + "\n"

            for fact in winds_morning_afternoon:
                context_facts += fact + "\n"

            context_facts += "\n"

            # for fact in cloud_moving_stripped:
            #    context_facts+=fact + "\n"
            # context_facts+="\n"

            for fact in temp_fronts:
                context_facts += fact + "\n"
            context_facts += "\n"
            for fact in hum_fronts:
                context_facts += fact + "\n"
            context_facts += "\n"
            '''
            context_facts+="% hum(location_1,humidity_percentage,hh): the percentage is discretized in 0, 20, 40 ... \n"
            for fact in hum_stripped:
                context_facts+=fact + "\n"
            context_facts+="\n"


            context_facts+="% Temperature data (in kelvin), last parameter is the hour:\n"
            for fact in temp_front_stripped:
                context_facts+=fact + "\n"
            context_facts+="\n"
            '''
            context_facts += "}). \n"

            with open(output_path, 'w') as f_out:
                f_out.write(positive_facts)
                f_out.write(excluded_facts)
                f_out.write(context_facts)

            print(f"Wrote example data to {output_path}")
            i += 1
        # copy in output_folder+"/bg.las" the file output_folder+"/../bg.las"

    shutil.copyfile("./reasoning/bg.las", output_folder + "/bg.las")


def compute_negative_facts(line, season):
    RAIN_VALUES = [0, 1, 2, 4, 6]
    match = fact_pattern.search(line)
    if not match:
        return ""  # not a matching forecast fact

    pred = match.group("pred")
    city = match.group("city")
    raw_value = match.group("value").strip()
    if (line.startswith('%')):
        return ""  # skip comments
    # Case 1: forecasted_rain(..., number)
    if pred == "forecasted_rain":
        try:
            value = int(raw_value)
        except ValueError:
            return "unkown_rain_at(" + city + "," + str(season) + "), \n"

        negatives = [
            f"rains_at({city},{v},{season})"
            for v in RAIN_VALUES
            if v != value
        ]

        return ", \n".join(negatives) + ", \n"

    # Case 2: forecasted_sky(..., "string")
    else:
        # Remove surrounding quotes if present
        if raw_value.startswith('"') and raw_value.endswith('"'):
            value = raw_value[1:-1]
        else:
            value = raw_value

        if value == "sunny" or value == "mostly_clear":
            return "partially_sunny_at(" + city + "," + str(season) + "), \n" + "covered_at(" + city + "," + str(
                season) + "), \n"
        elif value == "partly_cloudy":
            return "sunny_at(" + city + "," + str(season) + "), \n" + "covered_at(" + city + "," + str(season) + "), \n"
        elif value == "mostly_cloudy" or value == "cloudy":
            return "sunny_at(" + city + "," + str(season) + "), \n" + "partially_sunny_at(" + city + "," + str(
                season) + "), \n"
        else:
            return "unkown_sky_at(" + city + "," + str(season) + "), \n"

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
    load_locations(coordinates, base_path)

    for level, suff in folders_suff.items():

        entry = folder_types[1] + suff
        full_path = os.path.join(base_path, entry)

        print("starting date humidity: ", entry)
        # from height to hpa
        match = re.search(r'(\d+(?:[_\.]\d+)?(?:m|km))', entry)
        hpa = str(match.group(1))

        legend_colors, legend_values = load_legend_mapping(
            f"./raw_data/extracted_fvg_cleaned/legend_at_{hpa}_humidity.png")

        if os.path.isdir(full_path):

            # later this first stage can be skipped
            print("plotting locations to map")
            plot_locations_to_map(full_path, "reasoning/humidity/" + entry, coordinates)

            hum_values = get_humidity_over_locations_color(locations_name_px_pos, full_path, legend_colors,
                                                           legend_values)

            path = "reasoning/humidity/" + full_path.rsplit('/', 1)[-1] + "/humidity.txt"

            png_files = get_sorted_png_files(full_path)
            with open(path, "w") as f:
                f.write("% format humidity_percentage_at(location, humidity_percentage, yyyy, mm, dd, h).\n\n")
                for frame, values in hum_values.items():
                    filename = png_files[frame]
                    yyyy, mm, dd, h = extract_timestamp_from_filename(filename)
                    for location_name, _ in locations_name_px_pos.items():
                        loc_lower = location_name.lower().replace(" ", "_")
                        # witout appproximation use: int(values[location_name])
                        approximated_hum = int(round(values[location_name] / 20) * 20)  # approx to the nearest 10
                        f.write(f"humidity_percentage{suff}({loc_lower}, {approximated_hum}, {yyyy},{mm},{dd},{h}).\n")


def get_humidity_over_locations_color(locations, frames_path, legend_colors, legend_values, radius=5):
    """
    Given city positions and a folder of humidity frames,
    computes average humidity percentage around each location.
    """
    import os
    frame_humidity_map = {}
    frames = sorted([f for f in os.listdir(frames_path) if f.lower().endswith((".png", ".jpg", ".tif"))])

    for frame_idx, frame_file in enumerate(frames):
        img = np.array(Image.open(os.path.join(frames_path, frame_file)).convert("RGB"))
        h, w, _ = img.shape
        loc_to_hum = {}

        for name, (px, py) in locations.items():
            px_i, py_i = int(round(px)), int(round(py))
            if 0 <= px_i < w and 0 <= py_i < h:
                x_min, x_max = max(0, px_i - radius), min(w, px_i + radius + 1)
                y_min, y_max = max(0, py_i - radius), min(h, py_i + radius + 1)
                region = img[y_min:y_max, x_min:x_max]

                mean_color = region.reshape(-1, 3).mean(axis=0)
                hum = color_to_humidity(mean_color, legend_colors, legend_values)
                loc_to_hum[name] = hum
            else:
                print(f"{name} outside frame {frame_idx}")
        frame_humidity_map[frame_idx] = loc_to_hum
    return frame_humidity_map


def load_legend_mapping(legend_path, n_samples=101, min_value=0, max_value=100):
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
    load_locations(coordinates, base_path)

    for level, suff in folders_suff.items():

        entry = folder_types[2] + suff
        full_path = os.path.join(base_path, entry)

        # from height to hpa
        match = re.search(r'(\d+(?:[_\.]\d+)?(?:m|km))', entry)
        hpa = str(match.group(1))

        # open the file with min and max temp
        with open(f"./raw_data/extracted_fvg_cleaned/temp{folders_suff[level]}/legend{folders_suff[level]}_temp.txt",
                  'r') as ftxt:
            content = ftxt.read()

        match = re.search(r"Temperature range at \d+\s* hPa:\s*([\d.]+)\s*K\s*to\s*([\d.]+)\s*K", content)
        min_temp = None
        max_temp = None
        if match:
            min_temp = int(float(match.group(1)))
            max_temp = int(float(match.group(2)))
            print("temperatures:    ", min_temp, max_temp)
        else:
            raise ValueError("No temperature range found in legend file.")

        sample_points = max_temp - min_temp + 1  # 1 step each degree
        legend_colors, legend_values = load_legend_mapping(f"./raw_data/extracted_fvg_cleaned/legend_at_{hpa}_temp.png",
                                                           n_samples=sample_points, min_value=min_temp,
                                                           max_value=max_temp)

        if os.path.isdir(full_path):

            # later this first stage can be skipped
            plot_locations_to_map(full_path, "reasoning/temp/" + entry, coordinates)
            print("getting humidity over locations color FULL PATH", full_path)

            hum_values = get_humidity_over_locations_color(locations_name_px_pos, full_path, legend_colors,
                                                           legend_values)
            path = "reasoning/temp/" + full_path.rsplit('/', 1)[-1] + "/temp.txt"

            png_files = get_sorted_png_files(full_path)
            with open(path, "w") as f:
                f.write("% format temperature_at(location, temperature, yyyy, mm, dd, h).\n\n")
                for frame, values in hum_values.items():
                    filename = png_files[frame]
                    yyyy, mm, dd, h = extract_timestamp_from_filename(filename)
                    for location_name, _ in locations_name_px_pos.items():
                        loc_lower = location_name.lower().replace(" ", "_")
                        # witout appproximation use: int(values[location_name])
                        approximated_temp = round(values[location_name])  # approx (should already be int)
                        f.write(f"temperature_at{suff}({loc_lower}, {approximated_temp}, {yyyy},{mm},{dd},{h}).\n")
    return starting_date
