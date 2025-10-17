import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import xarray as xr
import scipy.ndimage as ndimage

# --- Dummy single image data ---
nx, ny = 100, 100
dx = dy = 3000  # meters per pixel
data = np.random.rand(1, ny, nx)  # time=1
x = np.arange(nx) * dx
y = -np.arange(ny) * dy  # negative for top-down
test_data = xr.DataArray(data, dims=("time","y","x"), coords={"time":[0], "y":y, "x":x})

# --- Smooth the frame ---
smoothed_frame = ndimage.gaussian_filter(test_data.isel(time=0).values, sigma=2)
temp_da = test_data.isel(time=0).copy()
temp_da.data = smoothed_frame

# --- Plot ---

fig, ax = plt.subplots(figsize=(6,6))
frame = temp_da.values
ax.imshow(frame, origin="upper", cmap="viridis")  # pixels are axes
ax.add_patch(patches.Circle((50,50), radius=5, edgecolor="red", facecolor="red", alpha=0.5))


# Label radius

ax.set_aspect("equal")
ax.set_title("Single Cloud Image with Circle")
plt.savefig("single_image_circle_corrected.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("✅ Plot saved as single_image_circle_corrected.png")
