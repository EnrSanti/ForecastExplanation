from image_processing.generateImage import generate_clustered_images, generate_heatMap, annotate_clouds_features
from GRIB.extract_features_nc import save_feature_maps
from GRIB.cut_long_lat import cut_grib_long_lat
import sys, os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
cropped_dir = "image_processing/cropped"                              
heatMap_dir = "image_processing/heatmaps"                           
clustered_dir = "image_processing/clustered"

feature_extraction_dir = "image_processing/ellipses"
numClusters = 3 



def extract(grib_file,coordinates_fvg,coordinates_italy, is_fvg):
    grib_path = os.path.join(input_dir, grib_file)
    base_name = os.path.splitext(grib_file)[0]  #remove .grib
    output_path = os.path.join(output_dir, base_name + "_cut.nc")
    
    #if already cut skip
    if not os.path.exists(output_path):
        print(f"CUTTING CUT: {grib_path}")
        #always cut the big chunk
        cut_grib_long_lat(grib_path, output_path, coordinates_italy)
        print(f"GRIB CUT: {output_path}")
    else:
        print(f"ALREADY CUT: {output_path}")

    print(f"EXTRACTING FEATURES: {output_path}")
    
    if(is_fvg):
        save_feature_maps(output_path, coordinates_fvg,True)
    else:
        save_feature_maps(output_path, coordinates_italy,False)

    print(f"Processed: {grib_file} → {output_path}")

if __name__ == "__main__":
    print("\n-------------------------------------------------\n[0] Info\n")

    print("-------------- From GRIB to images --------------")
    print("[1]: CUT Girb & extract FVG DATA")
    print("[2]: CUT Girb & extract IT DATA")
    
    print("-------------- Image processing -----------------")
    print("[3]: Generate heatmap images")
    print("[4]: Generate clusters images")
    print("[5]: Generate ellipses feature location")
    print("\nselect: ",end='')

    mode = int(input())
    if mode == 0:
        text = """
    The first two commands (1,2) cut (in latitude and long.) the gribs file under ./GRIB/data/original_CERRA, save it as .nc.
    Command 1 then extracts feature maps for the FVG region and stores them in "./GRIB/extracted_fvg".
    Command 2 extracts feature maps for the whole itealy and stores them in "./GRIB/extracted_it". 

    The remaining three commands (3,4,5) work on data under "./image_processing/":
        [3] Generates heatmaps from images in "./image_processing/cropped" storing them in "./image_processing/heatmaps"
        [4] Generates images clustered with 3 colors from the heatmaps (saving them in "./image_processing/clustered")
        [5] From the clustered images locates the main 'features' marking them with ellipses ("image_processing/ellipses"). 

    THE TWO PIPELINES ARE CURRENTLY DIVIDED (they work for now on different data)
                """
        print(text)

    elif mode == 1:
        #longmin longmax latmin latmax
        coordinates=[11.5,14.5,44.5,48]
        coordinates_italy=[6.5,18.5,36.5,48]
        input_dir = "./GRIB/data/original_CERRA"
        output_dir = "./GRIB/data/CERRA_cut"
        os.makedirs(output_dir, exist_ok=True)
        
        # all grib files
        grib_files = [f for f in os.listdir(input_dir) if f.endswith(".grib")]

        #no threads, processes HDF5 has some thread issues
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(extract, grib_file, coordinates,coordinates_italy,True): grib_file for grib_file in grib_files}

            for future in as_completed(futures):
                grib_file = futures[future]
                try:
                    future.result()  # raises exception if any
                except Exception as e:
                    print(f"Extract failed for {grib_file}: {e}")

    elif mode == 2:
        #longmin longmax latmin latmax
        coordinates=[11.5,14.5,44.5,48]
        coordinates_italy=[6.5,18.5,36.5,48]
        input_dir = "./GRIB/data/original_CERRA"
        output_dir = "./GRIB/data/CERRA_cut"
        os.makedirs(output_dir, exist_ok=True)
        
        # all grib files
        grib_files = [f for f in os.listdir(input_dir) if f.endswith(".grib")]

        #no threads, processes HDF5 has some thread issues
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(extract, grib_file, coordinates,coordinates_italy,False): grib_file for grib_file in grib_files}

            for future in as_completed(futures):
                grib_file = futures[future]
                try:
                    future.result()  # raises exception if any
                except Exception as e:
                    print(f"Extract failed for {grib_file}: {e}")


    elif mode == 3:    
        print("Generating heatmaps... (check folder " + heatMap_dir + ")")
        generate_heatMap(cropped_dir, heatMap_dir)
    elif mode == 4:
        print("Generating clusters... (check folder " + clustered_dir + ")")
        generate_clustered_images(numClusters, heatMap_dir, clustered_dir)
    elif mode == 5:
        print("Annotating features... (check folder " + feature_extraction_dir + ")")
        annotate_clouds_features(clustered_dir, feature_extraction_dir)
