from image_processing.image_proc import generate_clustered_images
from image_processing.image_proc import resize_1_4
from image_processing.segment_track import run_tobac
from GRIB.extract_features_nc import save_feature_maps
from GRIB.cut_long_lat import cut_grib_long_lat
import iris
iris.FUTURE.date_microseconds = True
import sys, os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings(
    "ignore",
    message="As of v1.6.0, segmentation with time length 1",
    category=UserWarning
)
numClusters = 3 
n_min_threshold=200



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
        save_feature_maps(output_path, coordinates_fvg,True,True)
    else:
        save_feature_maps(output_path, coordinates_italy,False,True)

    print(f"Processed: {grib_file} → {output_path}")

if __name__ == "__main__":
    print("\n-------------------------------------------------\n[0] Info\n")
        #generate_clustered_images(numClusters, heatMap_dir, clustered_dir)

    print("-------------- From GRIB to images --------------")
    print("[1]: CUT Girb & extract FVG DATA")
    print("[2]: CUT Girb & extract IT DATA")
    
    print("-------------- Image processing -----------------")
    print("[3]: Cluster & run TOBAC on FVG clustered data")
    print("[4]: run TOBAC on FVG data")
    print("[5]: Cluster & run TOBAC on IT clustered data")
    print("[6]: run TOBAC on IT data")

    print("\nselect: ",end='')

    mode = int(input())

    folders_pref = {"cloud"}#, "humidity"}#, "temp", "winds"}
    folders_suff = {
        1000: "_at_100m", 
        925: "_at_750m",
        850: "_at_1.4km",
        700: "_at_3km", 
        500: "_at_5.5km", 
        300: "_at_9km"
    }



    if mode == 0:
        text = """
    The first two commands (1,2) cut (in latitude and long.) the gribs file under ./GRIB/data/original_CERRA, save it as .nc.
    Command 1 then extracts feature maps for the FVG region and stores them in "./GRIB/extracted_fvg".
    Command 2 extracts feature maps for the whole itealy and stores them in "./GRIB/extracted_it". 

    The remaining three commands (3,4,5,6) work on data under "./image_processing/":
        [3] clusters the FVG images and runs TOBAC on them
        [4] scales FVG images and runs TOBAC on them
        [5] cluster the IT images and runs TOBAC on them
        [6] scales IT images and runs TOBAC on them
             """
        print(text)

    elif mode == 1:
        #longmin longmax latmin latmax
        coordinates=[11,15,44.5,48]
        coordinates_italy=[6.5,18.5,36.5,48]
        input_dir = "./GRIB/data/original_CERRA"
        output_dir = "./GRIB/data/CERRA_cut"
        os.makedirs(output_dir, exist_ok=True)
        
        # all grib files
        grib_files = [f for f in os.listdir(input_dir) if f.endswith(".grib")]

        #no threads, processes HDF5 has some thread issues
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(extract, grib_file, coordinates,coordinates_italy,True): grib_file for grib_file in grib_files}

            for future in as_completed(futures):
                grib_file = futures[future]
                try:
                    future.result()  # raises exception if any
                except Exception as e:
                    print(f"Extract failed for {grib_file}: {e}")

    elif mode == 2:
        #longmin longmax latmin latmax
        coordinates=[11,15,44.5,48]
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
        print("Cluster & run TOBAC on FVG clustered data (just clouds for now)")
            #resize_1_4("GRIB/extracted_fvg_cleaned/cloud_at_3km", "image_processing/fvg/resized")
        folder_list = [pref + suff for pref in folders_pref for suff in folders_suff.values()]
        for f in folder_list:
            resize_1_4(f"GRIB/extracted_fvg_cleaned/{f}", f"image_processing/fvg/resized/{f}")
            generate_clustered_images(numClusters, f"image_processing/fvg/resized/{f}", f"image_processing/fvg/clustered/{f}_clustered")
            run_tobac(f"image_processing/fvg/clustered/{f}_clustered", f"image_processing/fvg/output_clustered/{f}",n_min_threshold)

    elif mode == 4:
        print("run TOBAC on FVG data (just clouds for now)")
        folder_list = [pref + suff for pref in folders_pref for suff in folders_suff.values()]
        for f in folder_list:
            resize_1_4(f"GRIB/extracted_fvg_cleaned/{f}", f"image_processing/fvg/resized/{f}")
            run_tobac(f"image_processing/fvg/resized/{f}", f"image_processing/fvg/output/{f}",n_min_threshold)
    
    elif mode == 5:    
        
        print("Cluster & run TOBAC on IT clustered data (just clouds for now)")
            #resize_1_4("GRIB/extracted_fvg_cleaned/cloud_at_3km", "image_processing/fvg/resized")
        folder_list = [pref + suff for pref in folders_pref for suff in folders_suff.values()]
        for f in folder_list:
            resize_1_4(f"GRIB/extracted_it_cleaned/{f}", f"image_processing/it/resized/{f}")
            generate_clustered_images(numClusters, f"image_processing/it/resized/{f}", f"image_processing/it/clustered/{f}_clustered")
            run_tobac(f"image_processing/it/clustered/{f}_clustered", f"image_processing/it/output_clustered/{f}",n_min_threshold)

    elif mode == 6:
        print("run TOBAC on IT data (just clouds for now)")
        #resize_1_4("GRIB/extracted_fvg_cleaned/cloud_at_3km", "image_processing/fvg/resized")
        folder_list = [pref + suff for pref in folders_pref for suff in folders_suff.values()]
        for f in folder_list:
            resize_1_4(f"GRIB/extracted_it_cleaned/{f}", f"image_processing/it/resized/{f}")
            run_tobac(f"image_processing/it/resized/{f}", f"image_processing/it/output/{f}",n_min_threshold)
    