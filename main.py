from image_processing.generateImage import generate_clustered_images, generate_heatMap, annotate_clouds_features
from GRIB.extract_feature_maps import save_feature_maps

import sys, os

cropped_dir = "image_processing/cropped"                              
heatMap_dir = "image_processing/heatmaps"                           
clustered_dir = "image_processing/clustered"

feature_extraction_dir = "image_processing/ellipses"


numClusters = 3 

if __name__ == "__main__":
    print("\n-------------------------------------------------\n[0] Info\n")

    print("-------------- From GRIB to images --------------")
    print("[1]: Extract all FVG DATA")
    print("[2]: Extract all IT DATA\n")
    
    print("-------------- Image processing -----------------")
    print("[3]: Generate heatmap images")
    print("[4]: Generate clusters images")
    print("[5]: Generate ellipses feature location")
    print("\nselect: ",end='')

    mode = int(input())
    if mode == 0:
        text = """
    The first two commands (1,2) extract feature maps from a grib file under "./GRIB/data", 
    either from the whole map in the GRIB or from a specific region (NE Italy) 
    and store them in "./GRIB/extracted_it" or "./GRIB/extracted_fvg". 

    The remaining three commands (3,4,5) work on data under "./image_processing/":
        [3] Generates heatmaps from images in "./image_processing/cropped" storing them in "./image_processing/heatmaps"
        [4] Generates images clustered with 3 colors from the heatmaps (saving them in "./image_processing/clustered")
        [5] From the clustered images locates the main 'features' marking them with ellipses ("image_processing/ellipses"). 

    THE TWO PIPELINES ARE CURRENTLY DIVIDED (they work for now on different data)
                """
        print(text)

    elif mode == 1:
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
