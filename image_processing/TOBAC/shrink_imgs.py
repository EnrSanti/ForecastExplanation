import os
from PIL import Image

# === CONFIGURATION ===
input_folder = "to_test_real_2"         # your folder with original images
output_folder = os.path.join(input_folder, "resized")
scale_factor = 0.25             # shrink to 1/4 size

# === SETUP ===
os.makedirs(output_folder, exist_ok=True)

# Supported image formats
valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# === PROCESS IMAGES ===
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    name, ext = os.path.splitext(filename)

    if ext.lower() not in valid_extensions:
        continue  # skip non-image files

    try:
        # open image
        img = Image.open(file_path)

        # compute new size
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))

        # resize with high-quality downsampling
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

        # save to output folder
        output_path = os.path.join(output_folder, filename)
        img_resized.save(output_path)

        print(f"Resized: {filename} -> {new_size}")

    except Exception as e:
        print(f"Skipping {filename}: {e}")

