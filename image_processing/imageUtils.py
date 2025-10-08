import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def mask_color_black(img, low, high):
    low = np.array(low)
    high = np.array(high)

    # convert BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # create the Mask
    mask = cv2.inRange(imgHSV, low, high)
    # inverse mask
    mask = 255-mask
    res = cv2.bitwise_and(img, img, mask=mask)
    return res

def mask_color(img, low, high, mask_color):
    low = np.array(low)
    high = np.array(high)

    #BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(imgHSV, low, high)
    img[mask > 0] = mask_color
    return img



def recolor(fileNameImg, heatMapDir):
    # Load image
    img = cv2.imread(fileNameImg)

    # Convert to grayscale (intensity map)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement using histogram equalization
    enhanced = cv2.equalizeHist(gray)

    # Plot with Blues colormap
    plt.imshow(enhanced, cmap='Blues', vmin=0, vmax=255)
    plt.axis('off')

    # Get the original filename safely
    filename = os.path.basename(fileNameImg)
    output_path = os.path.join(heatMapDir, filename)

    # Save the result
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def cluster_images(n_im, numClusters, reshaped, image, image_f):
    clustering = [0 for _ in range(n_im)]
    for i in range(n_im):
        kmeans = KMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped[i])
        clustering[i] = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
                                   (image[i].shape[0], image[i].shape[1]))
        print("processing " + image_f[i])

    sortedLabels = [[] for _ in range(n_im)]
    for i in range(n_im):
        # compute mean brightness per cluster
        gray = cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)
        means = []
        for lbl in range(numClusters):
            mask = (clustering[i] == lbl)
            means.append(np.mean(gray[mask]) if np.any(mask) else 0)

        # sort by brightness (dark → bright)
        sortedLabels[i] = sorted(range(numClusters), key=lambda x: means[x])

    kmeansImage = [0 for _ in range(n_im)]
    concatImage = [[] for _ in range(n_im)]
    for j in range(n_im):
        kmeansImage[j] = np.zeros(image[j].shape[:2], dtype=np.uint8)
        for i, label in enumerate(sortedLabels[j]):
            # black = background, gray = border, white = core
            kmeansImage[j][clustering[j] == label] = int((255) / (numClusters - 1)) * i

        concatImage[j] = np.concatenate(
            (image[j],
             193 * np.ones((image[j].shape[0], int(0.0625 * image[j].shape[1]), 3), dtype=np.uint8),
             cv2.cvtColor(kmeansImage[j], cv2.COLOR_GRAY2BGR)),
            axis=1
        )

    return kmeansImage

