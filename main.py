from image_processing.generateImage import generate_clustered_images, generate_heatMap, annotate_clouds_features
from GRIB.extract_features_nc import save_feature_maps
from GRIB.cut_long_lat import cut_grib_long_lat
#from GRIB.extract_nc import save_feature_maps
import sys, os

cropped_dir = "image_processing/cropped"                              
heatMap_dir = "image_processing/heatmaps"                           
clustered_dir = "image_processing/clustered"

feature_extraction_dir = "image_processing/ellipses"


numClusters = 3 

if __name__ == "__main__":
    print("\n-------------------------------------------------\n[0] Info\n")

    print("-------------- From GRIB to images --------------")
    print("[1]: CUT Girb & extract FVG DATA")
    
    print("-------------- Image processing -----------------")
    print("[2]: Generate heatmap images")
    print("[3]: Generate clusters images")
    print("[4]: Generate ellipses feature location")
    print("\nselect: ",end='')

    mode = int(input())
    if mode == 0:
        text = """
    
    The first two command cuts (in latitude and long.) the grib file specified, saves it as .nc.
    It then extracts feature maps from it and stores them in "./GRIB/extracted_fvg". 

    The remaining three commands (2,3,4) work on data under "./image_processing/":
        [2] Generates heatmaps from images in "./image_processing/cropped" storing them in "./image_processing/heatmaps"
        [3] Generates images clustered with 3 colors from the heatmaps (saving them in "./image_processing/clustered")
        [4] From the clustered images locates the main 'features' marking them with ellipses ("image_processing/ellipses"). 

    THE TWO PIPELINES ARE CURRENTLY DIVIDED (they work for now on different data)
                """
        print(text)

    elif mode == 1:
        #longmin longmax latmin latmax
        coordinates=[11.5,14.5,44.5,48]
        grib_path = "./GRIB/data/2_9gb.grib"
        output_path = "./GRIB/data/2_9gb_cut.nc"
        cut_grib_long_lat(grib_path,output_path,coordinates)
        save_feature_maps(output_path,coordinates)
    elif mode == 2:    
        print("Generating heatmaps... (check folder " + heatMap_dir + ")")
        generate_heatMap(cropped_dir, heatMap_dir)
    elif mode == 3:
        print("Generating clusters... (check folder " + clustered_dir + ")")
        generate_clustered_images(numClusters, heatMap_dir, clustered_dir)
    elif mode == 4:
        print("Annotating features... (check folder " + feature_extraction_dir + ")")
        annotate_clouds_features(clustered_dir, feature_extraction_dir)
