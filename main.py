from image_processing.generateImage import generate_clustered_images, generate_heatMap, annotate_clouds_features

import sys, os

cropped_dir = "image_processing/cropped"                              
heatMap_dir = "image_processing/heatmaps"                           
clustered_dir = "image_processing/clustered"

feature_extraction_dir = "image_processing/ellipses"


numClusters = 3 

if __name__ == "__main__":
    print("[1]: Generate heatmap images")
    print("[2]: Generate clusters images")
    print("[3]: Generate ellipses feature location")

    mode = int(input())

    if mode == 1:
        print("Generating heatmaps...")
        generate_heatMap(cropped_dir, heatMap_dir)
    elif mode == 2:
        print("Generating clusters...")
        generate_clustered_images(numClusters, heatMap_dir, clustered_dir)
    elif mode == 3:
        print("Annotating features")
        annotate_clouds_features(clustered_dir, feature_extraction_dir)
