import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# -------------------------
# Example low-res field
# -------------------------
np.random.seed(0)
data = np.zeros((30, 30))
data[5:8, 5:8] = 1.0     # high value region (white-ish)
data[20:23, 20:23] = 0.3 # lower value region (light blue)
data += 0.02 * np.random.randn(*data.shape)

# -------------------------
# Extract contours
# -------------------------
# Choose contour levels. For low-res, a few levels are enough
levels = [0.3, 0.6, 0.9]

contours_list = []
for lvl in levels:
    contours = measure.find_contours(data, level=lvl)
    contours_list.append(contours)

# -------------------------
# Plot heatmap + contours
# -------------------------
plt.figure(figsize=(6,6))
plt.imshow(data, cmap="Blues", origin="lower")
colors = ["red", "yellow", "white"]  # one color per level

for color, contours in zip(colors, contours_list):
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color)

plt.title("Contour Extraction on Low-Res Data (Blues colormap)")
plt.axis("off")
plt.show()
