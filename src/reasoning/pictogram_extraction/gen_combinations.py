# LAUNCHED ONCE FROM BY LAUNCHING THE SCRIPT ITSELF NOT THE MAIN

import os

from PIL import Image

# Folder with your images
IMG_FOLDER = "./base_symbols"
OUTPUT_FOLDER = "./merged_icons"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Categories
categories = {
    "sun": ["sunny.png"],
    "cloud": ["big_cloud.png", "mid_cloud.png", "cloud.png", "small_cloud.png"],
    "rain": ["rain_1.png", "rain_2.png", "rain_4.png", "rain_6.png"],
    "snow": ["snow_1.png", "snow_2.png", "snow_3.png"],
    "misc": ["lightning.png"],
    # "exclusive": ["mist.png"]  # will not combine with anything
}


# Helper: merge images
def merge_images(img_list, output_name):
    base = None
    for img_file in img_list:
        img = Image.open(os.path.join(IMG_FOLDER, img_file)).convert("RGBA")
        if base is None:
            base = img
        else:
            base = Image.alpha_composite(base, img)
    base.save(os.path.join(OUTPUT_FOLDER, output_name))


# Generate valid combinations
def valid_combinations():
    result = []

    # Single images
    for cat, imgs in categories.items():
        for img in imgs:
            result.append([img])

    # 2-element combinations (only non-exclusive categories)
    cat1 = "sun"
    cat2 = "rain"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            result.append([img1, img2])
    cat2 = "snow"

    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            result.append([img1, img2])

    cat2 = "misc"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            result.append([img1, img2])

    cat1 = "cloud"
    cat2 = "rain"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            result.append([img1, img2])
    cat2 = "snow"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            result.append([img1, img2])

    cat2 = "misc"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            result.append([img1, img2])

    # 3-element combinations (only non-exclusive categories)
    cat1 = "sun"
    cat2 = "rain"
    cat3 = "snow"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            for img3 in categories[cat3]:
                result.append([img1, img2, img3])

    cat2 = "rain"
    cat3 = "misc"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            for img3 in categories[cat3]:
                result.append([img1, img2, img3])

    cat2 = "misc"
    cat3 = "snow"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            for img3 in categories[cat3]:
                result.append([img1, img2, img3])

    cat1 = "cloud"
    cat2 = "rain"
    cat3 = "snow"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            for img3 in categories[cat3]:
                result.append([img1, img2, img3])

    cat2 = "rain"
    cat3 = "misc"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            for img3 in categories[cat3]:
                result.append([img1, img2, img3])

    cat2 = "misc"
    cat3 = "snow"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            for img3 in categories[cat3]:
                result.append([img1, img2, img3])

    # 4-element combinations (only non-exclusive categories)
    cat1 = "sun"
    cat2 = "rain"
    cat3 = "snow"
    cat4 = "misc"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            for img3 in categories[cat3]:
                for img4 in categories[cat4]:
                    result.append([img1, img2, img3, img4])

    cat1 = "cloud"
    for img1 in categories[cat1]:
        for img2 in categories[cat2]:
            for img3 in categories[cat3]:
                for img4 in categories[cat4]:
                    result.append([img1, img2, img3, img4])

    return result


# Merge all combinations
for combo in valid_combinations():
    # If sun is in the combo, put it first (background)
    combo_sorted = sorted(combo, key=lambda x: 0 if x == "sunny.png" else 1)
    name = "_".join([c.replace(".png", "") for c in combo_sorted]) + ".png"
    merge_images(combo_sorted, name)
