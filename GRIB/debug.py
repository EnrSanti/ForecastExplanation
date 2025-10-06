from PIL import Image, ImageDraw
import pandas as pd

# --- Load image ---
image_path = "a.png"  # your PNG
img = Image.open(image_path)
draw = ImageDraw.Draw(img)

# --- Load CSV with vectors ---
csv_path = "./b.csv"
df = pd.read_csv(csv_path)

# --- Draw 5x5 px red squares ---
half_size = 2  # 5x5 square -> 2 pixels in each direction from center
for _, row in df.iterrows():
    x = int(row['pixel_x'])
    y = int(row['pixel_y'])
    bbox = [x - half_size, y - half_size, x + half_size, y + half_size]
    draw.rectangle(bbox, outline="red", fill="red")

# --- Save result ---
img.save("wind_image_with_vectors2.png")
