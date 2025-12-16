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
RELAXED_THRESHOLD = 0.80
SCALES_TO_TEST = [1]

# Filtering
OVERLAP_THRESHOLD = 0.5     # IoU threshold for NMS
LOCATION_TOLERANCE = 15     # pixels tolerance to match detected icon to CSV location

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
def resolve_perfect_matches(all_detections_raw, map_img_color, icon_templates):
    """
    Resolves near-perfect matches (score >= 0.99) at the same location/size by 
    prioritizing the template with the largest actual foreground area (most complex).
    """
    print("Resolving perfect matches using template complexity...")
    
    location_groups = {}
    for det in all_detections_raw:
        x, y, name, w, h, score = det
        key = (x, y, w, h)
        if key not in location_groups:
            location_groups[key] = []
        location_groups[key].append(det)

    filtered_detections = []

    for key, group in location_groups.items():
        # Find all detections in this group that scored near-perfectly
        perfect_matches = [det for det in group if det[5] >= 0.99]
        
        if len(perfect_matches) > 1:
            # --- Tie-Breaker Logic for Perfect Scores ---
            
            x, y, w, h = key
            
            # Patch extraction... (conversion to patch_match_channel remains the same)
            # Assuming you switched to a color channel (e.g., HSV Saturation)
            y_end, x_end = min(y + h, map_img_color.shape[0]), min(x + w, map_img_color.shape[1])
            patch = map_img_color[y:y_end, x:x_end]

            if patch.shape[0] != h or patch.shape[1] != w:
                 filtered_detections.extend(group) 
                 continue
            
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            patch_match_channel = patch_hsv[:, :, 1] 

            best_match = None
            # Store the highest value as a tuple: (custom_score, template_complexity_area)
            highest_custom_value = 0
            
            for det in perfect_matches:
                name = det[2]
            
                
                # --- Complexity Tie-Breaker ---
                # Retrieve the pre-calculated complexity area from the template structure
                template_complexity_area = icon_templates[name]["template_area"] 

                # Combine custom score (primary) and complexity (secondary)
                if template_complexity_area > highest_custom_value:
                    highest_custom_value = template_complexity_area
                    best_match = det
            
            # Keep only the single best match
            if best_match is not None:
                filtered_detections.append(best_match)
            
            # Add all non-0.99 detections back
            non_perfect_matches = [det for det in group if det[5] < 0.99]
            filtered_detections.extend(non_perfect_matches)

        else:
            # No tie, keep all detections in this group
            filtered_detections.extend(group)
            
    return filtered_detections
# --- Core Functions ---
def load_icon_with_mask(path, power=2.5):
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        raise FileNotFoundError(f"Could not load icon: {path}")

    # --- grayscale icon ---
    gray = cv2.cvtColor(icon[:, :, :3], cv2.COLOR_BGR2GRAY)

    if icon.shape[2] == 4:
        alpha = icon[:, :, 3]

        # binary alpha (cutout mask)
        _, cutout_mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

        # required by matchTemplate
        gray[alpha == 0] = 255

        # weighted bottom mask (for matching)

        template_complexity_area = np.sum(alpha > 0)
        return gray, cutout_mask, cutout_mask, template_complexity_area

    else:
        # no transparency
        return gray, None, None,0


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

    for location_name, det in detections.items():
        x, y, w, h = det["bbox"]
        name = det["type"]

        x, y, w, h = int(x), int(y), int(w), int(h)

        label = os.path.splitext(name)[0]
        if location_name:
            label += f" ({location_name})"

        cv2.rectangle(
            img_to_draw,
            (x, y),
            (x + w, y + h),
            color,
            thickness,
        )

        cv2.putText(
            img_to_draw,
            label,
            (x, max(y - 5, 10)),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, img_to_draw)
    print(f"Visualization saved to: {output_path}")



def visualize_detections_temp(image_path, detections, color=(0, 0, 255)):
    img_to_draw = cv2.imread(image_path)

    if img_to_draw is None:
        print(f"Error: Could not load image for visualization at {image_path}")
        return

    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    dets=detections["-"]
    for det in dets:
        
        x, y, w, h = det["bbox"]
        name = det["type"]
        score=det["score"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        cv2.rectangle(
            img_to_draw,
            (x, y),
            (x + w, y + h),
            color,
            thickness,
        )

        cv2.putText(
            img_to_draw,
            name+str(f" {score:.2f}"),
            (x, max(y - 5, 10)),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def build_bottom_weight_mask(h, w, power=2.5):
    y = np.linspace(0, 1, h).reshape(-1, 1)
    weights = y ** power
    mask = (weights * 255).astype(np.uint8)
    return np.repeat(mask, w, axis=1)

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
        return {}, []

    gray_map = cv2.cvtColor(map_img_color, cv2.COLOR_BGR2GRAY)
    all_detections_raw = []

    for name, data in icon_templates.items():
        icon = data["icon"]
        match_mask = data["match_mask"]

        for scale in SCALES_TO_TEST:
            h, w = icon.shape
            rw, rh = int(w * scale), int(h * scale)

            if rh > gray_map.shape[0] or rw > gray_map.shape[1]:
                continue

            resized_icon = cv2.resize(icon, (rw, rh))

            resized_mask = None
            if match_mask is not None:
                resized_mask = cv2.resize(match_mask, (rw, rh))

            res = cv2.matchTemplate(
                gray_map,
                resized_icon,
                cv2.TM_CCOEFF_NORMED,
                mask=resized_mask
            )

            res_max = maximum_filter(res, size=3)
            locs = np.where((res == res_max) & (res >= RELAXED_THRESHOLD))

            for y, x in zip(*locs):
                score = res[y, x]
                all_detections_raw.append((x, y, name, rw, rh, score))

    #remove -inf detections
    all_detections_raw = [det for det in all_detections_raw if np.isfinite(det[5])]
    all_detections_raw = resolve_perfect_matches(all_detections_raw, map_img_color, icon_templates)

    print(f"in image:  ----  { image_path.replace('_final', '_raw')} " )

    print_detections = {"-": []}
    for (x, y, name, w, h, score) in all_detections_raw:
        if((x>350 and x<370) and (y>205 and y<225) or not np.isfinite(score)):
            continue
        print(f" Detected: {name} at ({x}, {y}), size=({w}x{h}), score={score:.2f}")
            
        print_detections["-"].append({
            "type": name,
            "bbox": (x, y, w, h),
            "icon": name,
            "score": score
        })

    print(f"visualizing {len(print_detections)}  detections... ")
    #get the filename from image_path
    visualize_detections_temp(image_path, print_detections)
    # ---- NMS ----

    detections_to_filter = sorted(all_detections_raw, key=lambda x: x[5], reverse=True)
    final_detections = []
    suppressed = np.zeros(len(detections_to_filter), dtype=bool)

    #IOU
    for i in range(len(detections_to_filter)):
        if suppressed[i]:
            continue

        best = detections_to_filter[i]
        final_detections.append(best)
        rect_i = best[0], best[1], best[3], best[4]

        for j in range(i + 1, len(detections_to_filter)):
            if suppressed[j]:
                continue
            rect_j = detections_to_filter[j][0], detections_to_filter[j][1], detections_to_filter[j][3], detections_to_filter[j][4]
            if get_iou(rect_i, rect_j) > OVERLAP_THRESHOLD:
                suppressed[j] = True

    # ---- LOCATION MATCH + CUTOUT ----
    detections_with_locations = {}
    locations_detected = []

    for (x, y, name, w, h, score) in final_detections:
        loc_name = match_to_location(x, y, locations_df)
        if loc_name is None:
            continue    

        locations_detected.append(loc_name)

        # ---- CUT OUT ICON FROM MAP ----
        roi = map_img_color[y:y+h, x:x+w]
        cutout_mask = icon_templates[name]["cutout_mask"]

        if cutout_mask is not None:
            alpha = cv2.resize(cutout_mask, (w, h))
            roi_rgba = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
            roi_rgba[:, :, 3] = alpha
        else:
            roi_rgba = roi

        detections_with_locations[loc_name] = {
            "type": name,
            "bbox": (x, y, w, h),
            "icon": roi_rgba
        }

    return detections_with_locations, locations_detected


def icon_name_to_rain_level(icon_name):
    match = re.search(r'rain_([1-4]|6)', icon_name)
    if match:
        return int(match.group(1))
    
    match = re.search(r'no_rain', icon_name)
    if match:
        return 0
    return "atom_ND"

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
    
    if "big_cloud" in name:
        return "mostly_cloudy"
    elif "mid_cloud" in name:
        return "partly_cloudy"
    elif "small_cloud" in name:
        return "mostly_clear"
    elif "cloud" in name:
        return "cloudy"
    elif "sunny" in name:
        return "sunny"
    return "ND"

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
            gray, match_mask, cutout_mask, template_complexity_area = load_icon_with_mask(path)

            icon_templates[name] = {
                "icon": gray,
                "match_mask": match_mask,
                "cutout_mask": cutout_mask,
                "template_area": template_complexity_area
            }
        except Exception as e:
            print(f"Error loading {os.path.basename(path)}: {e}")

    # Process all images
    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.[pPjJgG][nNpP][gG]')))
    if not image_paths:
        print(f"No input images found in {INPUT_FOLDER}")
        exit(1)

    print(f"Found {len(image_paths)} images to process.\n")

    all_locations=locations_df["Location"].tolist() 
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing: {base_name}...")

        detections,locations_detected = perform_multi_scale_matching(img_path, icon_templates, locations_df)

        out_prefix = os.path.join(OUTPUT_FOLDER, base_name)
        txt_path = f"{out_prefix}_locations.txt"
        final_img_path = f"{out_prefix}_final.png"

        # Save text results with location names (no .png)

        yyyy,mm,dd=extract_date_from_filename(img_path)
        with open(txt_path, 'w') as f:
            f.write(f"%% Format: forecasted_rain(location, drops_in_pictogram).\n")
            f.write(f"%% Format: forecasted_sky(location, description).\n")
            f.write(f'date({yyyy},{mm},{dd}).\n\n')

            for loc in all_locations:
                loc_lower = loc.lower().replace(" ", "_")
                if(loc in detections):
                    det= detections[loc]
                    x, y, w, h = det["bbox"]
                    name = det["type"]
                    icon_name = os.path.splitext(name)[0]
                    print(loc)
                    f.write(f'forecasted_rain({loc_lower}, {icon_name_to_rain_level(icon_name)}). \n') #,{yyyy},{mm},{dd}
                    f.write(f'forecasted_sky({loc_lower}, "{icon_name_to_sky_level(icon_name)}"). \n') #,{yyyy},{mm},{dd}
                else:
                    f.write(f'forecasted_rain({loc_lower}, atom_ND). \n') #,{yyyy},{mm},{dd}
                    f.write(f'forecasted_sky({loc_lower}, "ND"). \n') #,{yyyy},{mm},{dd}

        visualize_detections(img_path, detections, final_img_path)

        print(f"  Saved: {txt_path}, {final_img_path}\n")

    print(f"--- Done. Total time: {time.time() - start_total:.2f} s ---")


