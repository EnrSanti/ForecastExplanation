import cv2
import numpy as np
import os
import glob
import time
import pandas as pd
from scipy.ndimage import maximum_filter
import re


# --- Configuration ---
ICON_FOLDER_PATH = './reasoning/pictogram_extraction/merged_icons/'
INPUT_FOLDER = './reasoning/pictogram_extraction/pictograms/sky/'      # folder containing all input images
OUTPUT_FOLDER = './reasoning/pictogram_extraction/extracted'           # folder where results will be saved
CSV_PATH = './reasoning/locations.csv'                                 # path to your location CSV

# Matching Parameters
RELAXED_THRESHOLD = 0.71
SCALES_TO_TEST = np.linspace(0.8, 1.2, 9)

# Filtering
OVERLAP_THRESHOLD = 0.5     # IoU threshold for NMS
LOCATION_TOLERANCE = 10     # pixels tolerance to match detected icon to CSV location

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Core Functions ---

def load_icon(path):
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        raise FileNotFoundError(f"Could not load icon: {path}")
    if icon.shape[2] == 4:
        alpha = icon[:, :, 3]
        gray = cv2.cvtColor(icon[:, :, :3], cv2.COLOR_BGR2GRAY)
        gray[alpha == 0] = 255  # fill transparent areas
    else:
        gray = cv2.cvtColor(icon[:, :, :3], cv2.COLOR_BGR2GRAY)
    return gray

def get_iou(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = x_overlap * y_overlap
    union_area = w1*h1 + w2*h2 - overlap_area
    return overlap_area / union_area if union_area > 0 else 0

def visualize_detections(image_path, detections, output_path, color=(0, 0, 255)):
    img_to_draw = cv2.imread(image_path)
    if img_to_draw is None:
        print(f"Error: Could not load image for visualization at {image_path}")
        return

    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35

    for x, y, name, w, h, location_name in detections:
        label = os.path.splitext(name)[0]  # remove extension
        if location_name:
            label += f" ({location_name})"
        cv2.rectangle(img_to_draw, (x, y), (x + w, y + h), color, 1)
        cv2.putText(img_to_draw, label, (x, y - 5), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, img_to_draw)
    print(f"Visualization saved to: {output_path}")

def match_to_location(x, y, locations_df):
    """Find the closest location within LOCATION_TOLERANCE pixels."""
    dx = locations_df['pictogram_px_x'] - x
    dy = locations_df['pictogram_px_y'] - y
    dist = np.sqrt(dx**2 + dy**2)
    min_idx = dist.idxmin()
    if dist[min_idx] <= LOCATION_TOLERANCE:
        return locations_df.loc[min_idx, 'Location']
    return None

def perform_multi_scale_matching(image_path, icon_templates, locations_df):
    map_img_color = cv2.imread(image_path)
    if map_img_color is None:
        print(f"!!! ERROR: Failed to load image: {image_path}")
        return []

    gray_map = cv2.cvtColor(map_img_color, cv2.COLOR_BGR2GRAY)
    all_detections_raw = []

    for name, icon in icon_templates.items():
        for scale in SCALES_TO_TEST:
            h, w = icon.shape
            resized = cv2.resize(icon, (int(w * scale), int(h * scale)))
            if resized.shape[0] > gray_map.shape[0] or resized.shape[1] > gray_map.shape[1]:
                continue

            res = cv2.matchTemplate(gray_map, resized, cv2.TM_CCOEFF_NORMED)
            res_max = maximum_filter(res, size=3)
            locs = np.where((res == res_max) & (res >= RELAXED_THRESHOLD))
            for y, x in zip(*locs):
                score = res[y, x]
                all_detections_raw.append((x, y, name, resized.shape[1], resized.shape[0], score))

    print(f"  Raw detections: {len(all_detections_raw)}")

    # IoU-based NMS
    detections_to_filter = sorted(all_detections_raw, key=lambda x: x[5], reverse=True)
    final_detections = []
    suppressed = np.zeros(len(detections_to_filter), dtype=bool)

    for i in range(len(detections_to_filter)):
        if suppressed[i]:
            continue
        best_match = detections_to_filter[i]
        final_detections.append(best_match)
        rect_i = best_match[0], best_match[1], best_match[3], best_match[4]

        for j in range(i + 1, len(detections_to_filter)):
            if suppressed[j]:
                continue
            rect_j = detections_to_filter[j][0], detections_to_filter[j][1], detections_to_filter[j][3], detections_to_filter[j][4]
            if get_iou(rect_i, rect_j) > OVERLAP_THRESHOLD:
                suppressed[j] = True

    # Add matched locations
    detections_with_locations = []
    for (x, y, name, w, h, score) in final_detections:
        loc_name = match_to_location(x, y, locations_df)
        detections_with_locations.append((x, y, name, w, h, loc_name))

    return detections_with_locations

def icon_name_to_rain_level(icon_name):
    match = re.search(r'rain_([1-4]|6)', icon_name)
    if match:
        return int(match.group(1))
    return 0 # no rain

def icon_name_to_sky_level(icon_name):
    """
    Returns a sky level based on the cloud keyword in the icon name:
      - 'cloud'      -> 4
      - 'big_cloud'  -> 3
      - 'mid_cloud'  -> 2
      - 'small_cloud' -> 1
      - otherwise    -> 0
    Matching is case-insensitive.
    """
    name = icon_name.lower()

    if "cloud" in name:
        return 4
    elif "big_cloud" in name:
        return 3
    elif "mid_cloud" in name:
        return 2
    elif "small_cloud" in name:
        return 1
    else:
        return 0


def extract_date_from_filename(filename):
    """
    Extracts a date (year, month, day) from strings like 'pitt_2019_11_04.png'.
    Returns a tuple of integers: (year, month, day)
    If no date is found, returns None.
    """
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
    if match:
        year, month, day = map(int, match.groups())
        return year, month, day
    return None


def generate_ground_truth():
    start_total = time.time()

    # Load CSV
    locations_df = pd.read_csv(CSV_PATH)
    if not {'Location', 'pictogram_px_x', 'pictogram_px_y'}.issubset(locations_df.columns):
        raise ValueError("CSV must contain columns: Location, pictogram_px_x, pictogram_px_y")

    # Load icon templates
    icon_templates = {}
    template_paths = glob.glob(os.path.join(ICON_FOLDER_PATH, '*.[pPjJgG][nNpP][gG]'))
    if not template_paths:
        print(f"Error: No icon images found in {ICON_FOLDER_PATH}")
        exit(1)

    print(f"Loaded {len(template_paths)} icon templates...")
    for path in template_paths:
        try:
            name = os.path.basename(path)
            icon_templates[name] = load_icon(path)
        except Exception as e:
            print(f"Error loading {os.path.basename(path)}: {e}")

    # Process all images
    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.[pPjJgG][nNpP][gG]')))
    if not image_paths:
        print(f"No input images found in {INPUT_FOLDER}")
        exit(1)

    print(f"Found {len(image_paths)} images to process.\n")

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing: {base_name}...")

        detections = perform_multi_scale_matching(img_path, icon_templates, locations_df)

        out_prefix = os.path.join(OUTPUT_FOLDER, base_name)
        txt_path = f"{out_prefix}_locations.txt"
        final_img_path = f"{out_prefix}_final.png"

        # Save text results with location names (no .png)

        yyyy,mm,dd=extract_date_from_filename(img_path)
        with open(txt_path, 'w') as f:
            f.write(f'date({yyyy},{mm},{dd}).\n')
            for x, y, name, w, h, loc in detections:
                icon_name = os.path.splitext(name)[0]  # remove .png/.jpg
                if loc:
                    f.write(f'forecasted_rain("{loc}", {icon_name_to_rain_level(icon_name)},{yyyy},{mm},{dd}). \n')
                    f.write(f'forecasted_sky("{loc}", {icon_name_to_sky_level(icon_name)},{yyyy},{mm},{dd}). \n')
                else:
                    f.write(f'UNKNOWN {icon_name}\n')

        #i don't save images for now
        #visualize_detections(img_path, detections, final_img_path)
        print(f"  Saved: {txt_path}, {final_img_path}\n")

    print(f"--- Done. Total time: {time.time() - start_total:.2f} s ---")
