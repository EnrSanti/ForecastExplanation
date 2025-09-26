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
    palette = [(255,255,255), (240, 240, 240)]

    very_low_green = [25, 0, 72]
    low_green = [25, 55, 72]
    high_green = [102, 255, 255]

    very_low_blue = [109, 0, 50]
    low_blue = [109, 140, 50]
    high_blue = [130, 255, 255]

    img = cv2.imread(fileNameImg)
    img = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    res = mask_color(img, low_green, high_green, palette[0])
    res = mask_color(res, low_blue, high_blue, palette[0])
    res = mask_color(res, very_low_blue, high_blue, palette[1])
    res = mask_color(res, very_low_green, high_blue, palette[1])

    maskBorders = cv2.imread("image_processing/maskBorders.jpg")
    maskBorders = cv2.cvtColor(maskBorders , cv2.COLOR_BGR2GRAY)
    maskBorders = cv2.resize(maskBorders, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    res = cv2.inpaint(res, maskBorders, 2, cv2.INPAINT_NS)

    cv2.imwrite("image_processing/step2_to_remove.jpg", res)
    
    img = np.asarray(Image.open('image_processing/step2_to_remove.jpg'))
    lum_img = img[:,:,0]
    imgplot = plt.imshow(lum_img, cmap='PuRd')
    plt.axis('off')

    # Get the original filename safely
    filename = os.path.basename(fileNameImg)
    output_path = os.path.join(heatMapDir, filename)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to avoid memory issues
    os.remove("image_processing/step2_to_remove.jpg")


def cluster_images(n_im, numClusters, reshaped, image, image_f):
    clustering=[0 for _ in range(0, n_im)]
    for i in range(0, n_im):
        kmeans = KMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped[i])
        clustering[i-1] = np.reshape(np.array(kmeans.labels_, dtype=np.uint8), (image[i].shape[0], image[i].shape[1]))
        print("processing " + image_f[i])

    sortedLabels=[[] for _ in range(0, n_im)]
    for i in range(0, n_im):
        sortedLabels[i] = sorted([n for n in range(numClusters)],
            key=lambda x: -np.sum(clustering[i] == x))


    kmeansImage=[0 for _ in range(0, n_im)]
    concatImage=[[] for _ in range(0, n_im)]
    for j in range(0, n_im):
        kmeansImage[j] = np.zeros(image[j].shape[:2], dtype=np.uint8)
        for i, label in enumerate(sortedLabels[j]):
            kmeansImage[j][ clustering[j] == label ] = int((255) / (numClusters - 1)) * i
        concatImage[j] = np.concatenate((image[j],193 * np.ones((image[j].shape[0], int(0.0625 * image[j].shape[1]), 3), dtype=np.uint8),cv2.cvtColor(kmeansImage[j], cv2.COLOR_GRAY2BGR)), axis=1)

    return kmeansImage
