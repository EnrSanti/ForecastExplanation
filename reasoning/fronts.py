path = "./extracted_grib_bw/cloud_at_5_5km/cloud_500_20191102_1700.png"
path1 = "./extracted_grib_bw/winds_at_5_5km/wind_500_20191102_1700.txt"
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

filename = os.path.basename(path)

import matplotlib.pyplot as plt

img = Image.open(path)
img_array = np.array(img)
red_channel = img_array.astype(np.float32) / 255.0
red_channel = red_channel[:, :, 0]  # Extract the red channel
print(red_channel.shape)

# Read wind data
df = pd.read_csv(path1)
print(df.head())


def create_magnitude_image(df, red_channel, tag):
    label = 'magnitude'
    if tag == 'x' or tag == 'y':
        label = 'alpha_deg'

    # Create 2D image with magnitude at correct pixel positions
    height, width = red_channel.shape
    magnitude_image = np.zeros((height, width))

    # Place magnitude values at pixel coordinates
    for idx, row in df.iterrows():
        x = int(row['pixel_x'])
        y = int(row['pixel_y'])
        if 0 <= x < width and 0 <= y < height:
            val = row[label]
            if tag == 'x':
                val = np.cos(np.deg2rad(row['alpha_deg']))
            elif tag == 'y':
                val = np.sin(np.deg2rad(row['alpha_deg']))
            magnitude_image[y, x] = val

    # Inpainting for missing values using griddata
    x = np.arange(magnitude_image.shape[1])
    y = np.arange(magnitude_image.shape[0])
    xx, yy = np.meshgrid(x, y)

    mask = magnitude_image == 0
    if mask.any():
        magnitude_image[mask] = griddata(
            (xx[~mask], yy[~mask]),
            magnitude_image[~mask],
            (xx[mask], yy[mask]),
            method='linear'
        )

    if tag == 'magnitude':
        # Normalize magnitude image 0..1 (vento max 50 m/s)
        magnitude_image = (magnitude_image / 50.0).clip(0, 1)

    magnitude_image_upsampled = magnitude_image
    # Upsample 4x using ndimage zoom
    # magnitude_image_upsampled = zoom(magnitude_image, zoom=4, order=1)
    # magnitude_image_upsampled = magnitude_image_upsampled[:red_channel.shape[0], :red_channel.shape[1]]

    # Identify pixels still equal to 0 after griddata
    remaining_mask = ~np.isfinite(magnitude_image_upsampled)

    # Apply inpainting to fill remaining zeros
    magnitude_image_uint8 = (magnitude_image_upsampled * 255).astype(np.uint8)
    # plt.imshow(magnitude_image_uint8)
    # plt.show()

    remaining_mask_uint8 = remaining_mask.astype(np.uint8)
    magnitude_image_upsampled = cv2.inpaint(magnitude_image_uint8, remaining_mask_uint8, 3, cv2.INPAINT_NS).astype(
        np.float32) / 255.0

    magnitude_image_upsampled = gaussian_filter(magnitude_image_upsampled, sigma=100)

    return magnitude_image_upsampled


# Call the function with appropriate parameters
magnitude_wind = create_magnitude_image(df, red_channel, 'magnitude')
degx_wind = create_magnitude_image(df, red_channel, 'x')
degy_wind = create_magnitude_image(df, red_channel, 'y')

# Apply Gaussian filter
red_channel = gaussian_filter(red_channel, sigma=100)

# Calculate Sobel gradients
sobel_x = ndimage.sobel(red_channel, axis=1)
sobel_y = ndimage.sobel(red_channel, axis=0)

# Calculate magnitude
magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

# Normalize direction images
sobel_x_normalized = (sobel_x / magnitude)
sobel_y_normalized = (sobel_y / magnitude)

# Calculate dot product between Sobel gradients and wind gradients
dot_product = (sobel_x_normalized * degx_wind) + (sobel_y_normalized * degy_wind)

w = dot_product
thr_w = 0.1
w = ((w - thr_w) / (1 - thr_w))
w[w < thr_w] = 0
w[~np.isfinite(w)] = 0
front = w * magnitude

thr_front = 0.005
front[front < thr_front] = 0
front[front >= thr_front] = (front[front >= thr_front] - thr_front)

# Label connected components
labeled_array, num_features = ndimage.label(front > 0)

# Filter regions by PCA analysis
pca_filtered_regions = np.zeros_like(front)  # Create an RGB image

for region_id in range(1, num_features + 1):
    region_mask = labeled_array == region_id
    coords = np.argwhere(region_mask)

    if len(coords) > 1000:  # Minimum region size at least 1000 pixels
        # Compute PCA weighted by front values
        coords_centered = coords - coords.mean(axis=0)
        weights = front[region_mask]
        # weights = weights / weights.sum()  # Normalize weights
        cov_matrix = np.cov(coords_centered.T, aweights=weights)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvector_max = eigenvectors[:, idx[0]]
        eigenvector_max = eigenvector_max / np.linalg.norm(eigenvector_max)

        # Check elongation ratio: lambda1/lambda2 > 5
        if eigenvalues[1] > 1e-6:
            ratio = eigenvalues[0] / eigenvalues[1]
            print(ratio)
            if ratio >= 20:
                mean_weight = weights.sum()
                pca_filtered_regions[region_mask] = mean_weight

                # Draw eigenvector as line through centroid
                centroid = coords.mean(axis=0)
                length = 100
                p1 = (centroid - eigenvector_max * length).astype(int)
                p2 = (centroid + eigenvector_max * length).astype(int)

                # Draw line on the region
                y1, x1 = np.clip(p1, 0, np.array(front.shape) - 1)
                y2, x2 = np.clip(p2, 0, np.array(front.shape) - 1)
                rr, cc = np.linspace(y1, y2, 50).astype(int), np.linspace(x1, x2, 50).astype(int)
                valid = (rr >= 0) & (rr < front.shape[0]) & (cc >= 0) & (cc < front.shape[1])
                pca_filtered_regions[rr[valid], cc[valid]] = 1

front = pca_filtered_regions

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(magnitude, cmap='gray')
axes[0, 0].set_title('Magnitude (Sobel)')
axes[0, 0].axis('off')
axes[0, 1].imshow(sobel_x_normalized, cmap='gray', vmin=-1, vmax=1)
axes[0, 1].set_title('Gradient X (Sobel)')
axes[0, 1].axis('off')
axes[0, 2].imshow(sobel_y_normalized, cmap='gray', vmin=-1, vmax=1)
axes[0, 2].set_title('Gradient Y (Sobel)')
axes[0, 2].axis('off')
axes[1, 0].imshow(front, cmap='gray')
axes[1, 0].set_title('Dot Product (Sobel x Wind)')
axes[1, 0].axis('off')
axes[1, 1].imshow(red_channel, cmap='gray')
axes[1, 1].set_title('cloud')
axes[1, 1].axis('off')
axes[1, 2].imshow(degy_wind, cmap='gray', vmin=-1, vmax=1)
axes[1, 2].set_title('Gradient Y (Wind)')
axes[1, 2].axis('off')
plt.tight_layout()
plt.show()
