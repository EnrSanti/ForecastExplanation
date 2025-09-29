from image_processing.generateImage import generate_clustered_images, generate_heatMap, annotate_clouds_features
from GRIB.extract_feature_maps import save_feature_maps

import sys, os

cropped_dir = "image_processing/cropped"                              
heatMap_dir = "image_processing/heatmaps"                           
clustered_dir = "image_processing/clustered"

feature_extraction_dir = "image_processing/ellipses"


numClusters = 3 

if __name__ == "__main__":
    print("-------------- From GRIB to images --------------")
    print("[1]: Extract all FVG DATA")
    print("[2]: Extract all IT DATA")
    print("-------------- Image processing --------------")
    print("[3]: Generate heatmap images")
    print("[4]: Generate clusters images")
    print("[5]: Generate ellipses feature location")
    print("select: ",end='')

    mode = int(input())
    if mode == 1:
        #longmin longmax latmin latmax
        coordinates=(11.5,14.5,44.5,48)
        save_feature_maps(coordinates)
    elif mode == 2:
        #default, all map
        coordinates=(None,None,None,None)
        save_feature_maps(coordinates)
    elif mode == 3:    
        print("Generating heatmaps... (check folder " + heatMap_dir + ")")
        generate_heatMap(cropped_dir, heatMap_dir)
    elif mode == 4:
        print("Generating clusters... (check folder " + clustered_dir + ")")
        generate_clustered_images(numClusters, heatMap_dir, clustered_dir)
    elif mode == 5:
        print("Annotating features... (check folder " + feature_extraction_dir + ")")
        annotate_clouds_features(clustered_dir, feature_extraction_dir)
