from image_processing.image_proc import generate_clustered_images
from image_processing.image_proc import resize_1_4_and_simplify
from image_processing.segment_track import run_tobac
from raw_data.extract_features_nc import save_feature_maps
from raw_data.cut_long_lat import cut_grib_long_lat
import iris
iris.FUTURE.date_microseconds = True
import sys, os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from reasoning.generate_examples import coordinates,coordinates_italy
warnings.filterwarnings(
    "ignore",
    message="As of v1.6.0, segmentation with time length 1",
    category=UserWarning
)

#for clustering the number of clusters to consider
numClusters = 3 

#minimum number of pixels for TOBAC (don't consider smaller blobs)
n_min_threshold=300




def extract(grib_file,coordinates_fvg,coordinates_italy, is_fvg):

    """
    Extracts the a smaller .nc data file from the grib files and saves them with the proper name (date etc...). 
    It's used so that the all the rest of the pipeline works on smaller files.
    It then extracts the feature maps for the given coordinates (FVG or Italy).

    Parameters
    ----------
    grib_file: the file from which to extract the data
    coordinates_fvg: the coordinates to cut for FVG (the grib has a lot of extra data, bigger area)
    coordinates_italy: the coordinates to cut for Italy (the grib has a lot of extra data, bigger area)
    is_fvg: boolean, if true extracts for FVG, else for Italy
    
    """
    
    #get the file name and paths
    grib_path = os.path.join(input_dir, grib_file)
    base_name = os.path.splitext(grib_file)[0]  #remove .grib
    output_path = os.path.join(output_dir, base_name + "_cut.nc")
    
    #if already cut skip
    if not os.path.exists(output_path):
        #always cut the big chunk, we want to keep a single file, which is still smaller than the original grib
        cut_grib_long_lat(grib_path, output_path, coordinates_italy)
        print(f"GRIB CUT: {output_path}")
    else:
        print(f"ALREADY CUT: {output_path}")

    print(f"EXTRACTING FEATURES: {output_path}")
    
    #proper image extraction
    if(is_fvg):
        save_feature_maps(output_path, coordinates_fvg,True,True)
    else:
        save_feature_maps(output_path, coordinates_italy,False,True)

    print(f"Processed: {grib_file} -> {output_path}")




if __name__ == "__main__":
    print("\n-------------------------------------------------\n[0] Info\n")

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

    folders_pref = {"cloud"} #, "humidity"}#, "temp", "winds"}
    folders_suff = {
        1000: "_at_100m", 
        925: "_at_750m",
        850: "_at_1.4km",
        700: "_at_3km", 
        500: "_at_5.5km", 
        300: "_at_9km"
    }


    #print info

    if mode == 0:
        text = """
    The first two commands (1,2) cut (in latitude and long.) the gribs file under ./raw_data/data/original_CERRA, save it as .nc.
    Command 1 then extracts feature maps for the FVG region and stores them in "./raw_data/extracted_fvg".
    Command 2 extracts feature maps for the whole itealy and stores them in "./raw_data/extracted_it". 

    The remaining three commands (3,4,5,6) work on data under "./image_processing/":
        [3] clusters the FVG images and runs TOBAC on them
        [4] scales FVG images and runs TOBAC on them
        [5] cluster the IT images and runs TOBAC on them
        [6] scales IT images and runs TOBAC on them
             """
        print(text)

    elif mode == 1:
        #extract nc from grib for FVG & save feature maps (multithreaded, beware more threads use lots of RAM) 
        input_dir = "./raw_data/data/original_CERRA"
        output_dir = "./raw_data/data/CERRA_cut"
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
        #extract nc from grib for IT & save feature maps  (multithreaded, beware more threads use lots of RAM)
        coordinates=[11,15,44.5,48]
        coordinates_italy=[6.5,18.5,36.5,48]
        input_dir = "./raw_data/data/original_CERRA"
        output_dir = "./raw_data/data/CERRA_cut"
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
        #resize iamges once to work on smaller images, generate clustered images and run TOBAC (FVG)
        print("Cluster & run TOBAC on FVG clustered data (just clouds for now)")
        folder_list = [pref + suff for pref in folders_pref for suff in folders_suff.values()]
        for f in folder_list:
            resize_1_4_and_simplify(f"raw_data/extracted_fvg_cleaned/{f}", f"image_processing/fvg/resized/{f}")
            generate_clustered_images(numClusters, f"image_processing/fvg/resized/{f}", f"image_processing/fvg/clustered/{f}_clustered")
            run_tobac(f"image_processing/fvg/clustered/{f}_clustered", f"image_processing/fvg/output_clustered/{f}","raw_data/extracted_fvg_cleaned/borders.png",coordinates[2],coordinates[3],coordinates[0],coordinates[1],n_min_threshold)

    elif mode == 4:
        #resize iamges once to work on smaller images and run TOBAC (FVG)
        print("run TOBAC on FVG data (just clouds for now)")
        folder_list = [pref + suff for pref in folders_pref for suff in folders_suff.values()]
        for f in folder_list:
            resize_1_4_and_simplify(f"raw_data/extracted_fvg_cleaned/{f}", f"image_processing/fvg/resized/{f}")
            run_tobac(f"image_processing/fvg/resized/{f}", f"image_processing/fvg/output/{f}","raw_data/extracted_fvg_cleaned/borders.png",coordinates[2],coordinates[3],coordinates[0],coordinates[1],n_min_threshold)
    
    elif mode == 5:    
        #resize iamges once to work on smaller images, generate clustered images and run TOBAC (IT)
        print("Cluster & run TOBAC on IT clustered data (just clouds for now)")
        folder_list = [pref + suff for pref in folders_pref for suff in folders_suff.values()]
        for f in folder_list:
            resize_1_4_and_simplify(f"raw_data/extracted_it_cleaned/{f}", f"image_processing/it/resized/{f}")
            generate_clustered_images(numClusters, f"image_processing/it/resized/{f}", f"image_processing/it/clustered/{f}_clustered")
            run_tobac(f"image_processing/it/clustered/{f}_clustered", f"image_processing/it/output_clustered/{f}","raw_data/extracted_it_cleaned/borders.png",coordinates_italy[2],coordinates_italy[3],coordinates_italy[0],coordinates_italy[1],n_min_threshold)

    elif mode == 6:
        #resize iamges once to work on smaller images and run TOBAC (FVG)
        print("run TOBAC on IT data (just clouds for now)")
      
        folder_list = [pref + suff for pref in folders_pref for suff in folders_suff.values()]
        for f in folder_list:
            resize_1_4_and_simplify(f"raw_data/extracted_it_cleaned/{f}", f"image_processing/it/resized/{f}")
            run_tobac(f"image_processing/it/resized/{f}", f"image_processing/it/output/{f}","raw_data/extracted_it_cleaned/borders.png",coordinates_italy[2],coordinates_italy[3],coordinates_italy[0],coordinates_italy[1],n_min_threshold)
    