import glob
import os
import re
import time

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

# --- Configuration ---
ICON_FOLDER_PATH = './reasoning/pictogram_extraction/base_symbols/'
INPUT_FOLDER = './reasoning/pictogram_extraction/pictograms/sky/'  # folder containing all input images
OUTPUT_FOLDER = './reasoning/pictogram_extraction/extracted'  # folder where results will be saved
CSV_PATH = './reasoning/locations.csv'  # path to your location CSV

# Matching Parameters
RELAXED_THRESHOLD = 0.90
SCALES_TO_TEST = [1]

# Filtering
OVERLAP_THRESHOLD = 0.5  # IoU threshold for NMS
LOCATION_TOLERANCE = 30  # pixels tolerance to match detected icon to CSV location

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def resolve_perfect_matches(all_detections_raw, map_img_color, icon_templates, locations_df):
    """
    Resolves near-perfect matches (score >= 0.99) at the same location/size 
    grouped by CATEGORY.
    """
    print("Resolving perfect matches using template complexity (per category)...")

    # location_groups now uses (x, y, w, h, category) as a key
    location_groups = {}
    for det in all_detections_raw:
        # det is now (x, y, name, w, h, score, category)
        x, y, name, w, h, score, category = det
        location = match_to_location(x, y, locations_df)
        key = (location, category)
        if key not in location_groups:
            location_groups[key] = []
        location_groups[key].append(det)

    # group by ONLY LOCATION in to_print
    to_print = {}
    for key, group in location_groups.items():
        location, category = key
        if location not in to_print:
            to_print[location] = []
        to_print[location].append((category, group))

    for key, group in to_print.items():
        print(f"Key: {key}, Group Size: {len(group)}")
        print(f"  Detections: {group}\n")

    filtered_detections = []
    for key, group in location_groups.items():
        # Find all detections in this group that scored near-perfectly
        perfect_matches = [det for det in group if det[5] >= 0.90]
        if len(perfect_matches) > 1:
            location, category = key

            # Tie-Breaker Logic
            best_match = None
            highest_complexity = -1

            for det in perfect_matches:
                name = det[2]
                cat = det[6]  # category

                # Access the nested template dictionary
                # icon_templates[cat][name]
                template_complexity_area = icon_templates[cat][name]["template_area"]
                if template_complexity_area > highest_complexity:
                    highest_complexity = template_complexity_area
                    best_match = det

            if best_match is not None:
                filtered_detections.append(best_match)


        else:
            # No tie within this category/location, keep it
            filtered_detections.extend(group)

    return filtered_detections


# --- Core Functions ---
def load_icon_with_mask(path):
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
        return gray, None, None, 0


def get_iou(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = x_overlap * y_overlap
    union_area = w1 * h1 + w2 * h2 - overlap_area
    return overlap_area / union_area if union_area > 0 else 0


def visualize_detections(image_path, detections, output_path, color=(0, 0, 255)):
    img_to_draw = cv2.imread(image_path)

    if img_to_draw is None:
        print(f"Error: Could not load image for visualization at {image_path}")
        return

    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3  # Slightly smaller to avoid overlap

    # Colors for different categories to help distinguish them
    cat_colors = {
        "sky": (0, 0, 255),  # Red
        "rain": (255, 0, 0),  # Blue
        "snow": (255, 255, 0),  # Cyan
        "thunder": (0, 255, 255)  # Yellow
    }

    # detections is: { location_name: { category: {type, score, bbox}, ... }, ... }
    for location_name, categories in detections.items():
        # Iterate through each component detected at this location
        for cat_name, det_data in categories.items():
            x, y, w, h = det_data["bbox"]
            icon_file = det_data["type"]
            score = det_data["score"]

            x, y, w, h = int(x), int(y), int(w), int(h)

            # Use specific color for category if defined, else default
            draw_color = cat_colors.get(cat_name, color)

            # Create a label showing Category, Icon Name, and Score
            clean_name = os.path.splitext(icon_file)[0]
            label = f"{location_name}|{cat_name}: {clean_name} ({score:.2f})"

            # Draw the box for this specific component
            cv2.rectangle(
                img_to_draw,
                (x, y),
                (x + w, y + h),
                draw_color,
                thickness,
            )

            # Draw the text (offset y slightly for each category to prevent stacking text)
            # Sky at top, Rain slightly lower, etc.
            y_offset = 0
            if cat_name == "rain":
                y_offset = 10
            elif cat_name == "snow":
                y_offset = 20
            elif cat_name == "thunder":
                y_offset = 30

            cv2.putText(
                img_to_draw,
                label,
                (x, max(y - 5 + y_offset, 10 + y_offset)),
                font,
                font_scale,
                draw_color,
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
    dets = detections["-"]
    for det in dets:
        x, y, w, h = det["bbox"]
        name = det["type"]
        score = det["score"]
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
            name + str(f" {score:.2f}"),
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
    dist = np.sqrt(dx ** 2 + dy ** 2)
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

    # We will collect all detections across all categories
    all_detections_raw = []

    # 1. Nested For: Category -> Individual Template
    for category, templates in icon_templates.items():

        for name, data in templates.items():
            icon = data["icon"]
            match_mask = data["match_mask"]

            for scale in SCALES_TO_TEST:
                h, w = icon.shape
                rw, rh = int(w * scale), int(h * scale)

                if rh > gray_map.shape[0] or rw > gray_map.shape[1]:
                    continue

                resized_icon = cv2.resize(icon, (rw, rh))
                resized_mask = cv2.resize(match_mask, (rw, rh)) if match_mask is not None else None

                method = cv2.TM_CCORR_NORMED if category == "snow" else cv2.TM_CCOEFF_NORMED

                res = cv2.matchTemplate(
                    gray_map,
                    resized_icon,
                    method,
                    mask=resized_mask
                )

                res_max = maximum_filter(res, size=3)
                # Use a slightly lower threshold if desired for small components like rain
                locs = np.where((res == res_max) & (res >= RELAXED_THRESHOLD))

                for y, x in zip(*locs):
                    score = res[y, x]
                    if np.isfinite(score):
                        # Added 'category' to the detection tuple for tracking
                        all_detections_raw.append((x, y, name, rw, rh, score, category))

    # remove -inf detections
    all_detections_raw = [det for det in all_detections_raw if np.isfinite(det[5])]

    all_detections_raw = resolve_perfect_matches(all_detections_raw, map_img_color, icon_templates, locations_df)

    print_detections = {"-": []}

    for (x, y, name, w, h, score, cat) in all_detections_raw:
        if ((x > 350 and x < 370) and (y > 205 and y < 225) or not np.isfinite(score)):
            continue

        print_detections["-"].append({
            "type": name,
            "bbox": (x, y, w, h),
            "icon": name,
            "score": score
        })

    print(f"visualizing {len(print_detections)}  detections... ")
    # get the filename from image_path
    visualize_detections_temp(image_path, print_detections)

    # -------------------------------------------------
    # Instead of simple NMS, we group by location and pick the best icon for EACH category
    detections_with_locations = {}

    for (x, y, name, w, h, score, cat) in all_detections_raw:
        loc_name = match_to_location(x, y, locations_df)
        if loc_name is None:
            continue

        if loc_name not in detections_with_locations:
            detections_with_locations[loc_name] = {}

        # Keep the best match for this specific category at this location
        if cat not in detections_with_locations[loc_name] or score > detections_with_locations[loc_name][cat]['score']:
            detections_with_locations[loc_name][cat] = {
                "type": name,
                "score": score,
                "bbox": (x, y, w, h)
            }

    # Note: This returns a dict of dicts: results[location][category]
    return detections_with_locations, list(detections_with_locations.keys())


def icon_name_to_rain_level(icon_name):
    match = re.search(r'rain_([1-4]|6)', icon_name)
    if match:
        return int(match.group(1))
    return 0
    return 0


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

    match = re.search(r'(\d{4})_(\d{1,2})_(\d{1,2})', filename)
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
    icon_templates = {"sky": {}, "rain": {}, "snow": {}, "thunder": {}}
    template_paths = glob.glob(os.path.join(ICON_FOLDER_PATH, '*.[pPjJgG][nNpP][gG]'))
    if not template_paths:
        print(f"Error: No icon images found in {ICON_FOLDER_PATH}")
        exit(1)

    print(f"Loaded {len(template_paths)} icon templates...")
    for path in template_paths:
        try:
            name = os.path.basename(path)
            gray, match_mask, cutout_mask, template_complexity_area = load_icon_with_mask(path)

            if ("rain" in name):
                icon_templates["rain"][name] = {
                    "icon": gray,
                    "match_mask": match_mask,
                    "cutout_mask": cutout_mask,
                    "template_area": template_complexity_area
                }
            elif ("snow" in name):
                icon_templates["snow"][name] = {
                    "icon": gray,
                    "match_mask": match_mask,
                    "cutout_mask": cutout_mask,
                    "template_area": template_complexity_area
                }
            elif ("lightning" in name):
                icon_templates["thunder"][name] = {
                    "icon": gray,
                    "match_mask": match_mask,
                    "cutout_mask": cutout_mask,
                    "template_area": template_complexity_area
                }
            else:
                icon_templates["sky"][name] = {
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

    all_locations = locations_df["Location"].tolist()
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing: {base_name}...")

        detections, _ = perform_multi_scale_matching(img_path, icon_templates, locations_df)

        out_prefix = os.path.join(OUTPUT_FOLDER, base_name)
        txt_path = f"{out_prefix}_locations.txt"
        final_img_path = f"{out_prefix}_final.png"

        # Save text results with location names (no .png)

        yyyy, mm, dd = extract_date_from_filename(img_path)
        with open(txt_path, 'w') as f:
            # f.write(f"%% Format: forecasted_rain(location, drops_in_pictogram).\n")
            # f.write(f"%% Format: forecasted_sky(location, description).\n")
            f.write(f'% date({yyyy},{mm},{dd}),\n\n')

            lines = []

            for loc in all_locations:
                loc_lower = loc.lower().replace(" ", "_")

                if loc in detections:
                    loc_data = detections[loc]  # This is a dict of categories

                    # 1. Get Sky (Coverage) - REQUIRED
                    sky_det = loc_data.get("sky")
                    if sky_det:
                        # Extract "big_cloud", "sunny", etc.
                        sky_val = os.path.splitext(sky_det["type"])[0]
                    else:
                        sky_val = "ND"

                    # 2. Extract specific levels for appendages
                    components = [sky_val]

                    # Rain
                    rain_det = loc_data.get("rain")
                    if rain_det:
                        r_name = os.path.splitext(rain_det["type"])[0]
                        # Assuming r_name is like 'rain_1', we keep it as is
                        components.append(r_name)

                    # Snow
                    snow_det = loc_data.get("snow")
                    if snow_det:
                        s_name = os.path.splitext(snow_det["type"])[0]
                        components.append(s_name)

                    # Lightning
                    '''SKIP FOR NOW
                    thunder_det = loc_data.get("thunder")
                    if thunder_det:
                        # Just add the word 'lightning' if detected
                        components.append("lightning")
                    '''
                    # Join them with underscores: e.g., "big_cloud_rain_1_snow_3_lightning"
                    full_icon_name = "_".join(components)

                    # Determine sky level description for the text file
                    # You might want to pass the joined name or just the sky component
                    sky_description = icon_name_to_sky_level(full_icon_name)

                    lines.append(
                        f'forecasted_sky({loc_lower}, "{sky_description}")'
                    )

                    # Optional: If you want to keep the rain level logic available
                    rain_lvl = icon_name_to_rain_level(full_icon_name)
                    lines.append(f'forecasted_rain({loc_lower}, {rain_lvl})')

                else:
                    lines.append(f'forecasted_sky({loc_lower}, "ND")')
            f.write(",\n".join(lines))

        visualize_detections(img_path, detections, final_img_path)

        print(f"  Saved: {txt_path}, {final_img_path}\n")

    print(f"--- Done. Total time: {time.time() - start_total:.2f} s ---")
