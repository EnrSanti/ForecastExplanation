from image_processing.image_proc import generate_clustered_images
from image_processing.image_proc import resize_1_4_and_simplify
from image_processing.segment_track import run_tobac_merge_split, run_tobac_fronts

from image_processing.humidity_front import get_humidity_front
from raw_data.extract_features_nc import save_feature_maps
from raw_data.cut_long_lat import cut_grib_long_lat
from reasoning.pictogram_extraction.pictograms_to_ground_truth import generate_ground_truth
from reasoning.generate_examples import generate_cloud_facts_over_cities, generate_humidity_facts_over_cities,generate_temp_facts_over_cities , merge_into_examples
from reasoning.get_fronts import init_fronts_generation,generate_fronts_hum,generate_fronts_temp

import iris
iris.FUTURE.date_microseconds = True
import sys, os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from reasoning.generate_examples import coordinates,coordinates_italy, init_starting_date

warnings.filterwarnings(
    "ignore",
    message="As of v1.6.0, segmentation with time length 1",
    category=UserWarning
)

#for clustering the number of clusters to consider
numClusters_clouds = 3
num_clusters_fronts = 5

#minimum number of pixels for TOBAC (don't consider smaller blobs)
n_min_threshold_clouds=300
n_min_threshold_fronts=2000




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
    
    print("-------------- Image processing -----------------")
    print("[2]: Cluster & run TOBAC on FVG clustered data")
    print("     Generate facts (coulds, humidty...) over each city")
    print("[3]: run TOBAC on FVG data")
    print("     Generate facts (coulds, humidty...) over each city")
    print("[4]: Extract ground truth from pictograms")
    print("[5]: Generate full examples (to complete)")

    print("\nselect: ",end='')

    mode = int(input())

    folders_pref = ["cloud","humidity","temp"] #, "winds"}
    
    folder_params = {
        "cloud": (0.7, "maximum",numClusters_clouds),       # e.g. (threshold, go lower or upper, clusters)
        "humidity": (0.55, "minimum",num_clusters_fronts),
        "temp": (0.6, "maximum",num_clusters_fronts),       # temp fronts
    }

    folders_suff = {
        1000: "_at_100m", 
        925: "_at_750m",
        850: "_at_1_4km",
        700: "_at_3km", 
        500: "_at_5_5km", 
        300: "_at_9km"
    }


    #print info

    if mode == 0:
        text = """
        The first two commands (1) cut (in latitude and long.) the gribs file under ./raw_data/data/original_CERRA, save it as .nc.
        Then extracts feature maps for the FVG region and stores them in "./raw_data/extracted_fvg".
        
        The remaining three commands (2,3) work on data under "./image_processing/":
            [2] clusters the FVG images and runs TOBAC on them
                Generate facts (coulds, humidty...) over each city
            [3] scales FVG images and runs TOBAC on them
                Generate facts (coulds, humidty...) over each city
            [4] extract ground truth from pictograms (put them under ./reasoning/pictogram_extraction/pictograms)
            [5]: Generate full examples (TODO)
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
        #resize iamges once to work on smaller images, generate clustered images and run TOBAC (FVG)
        print("Cluster & run TOBAC on FVG clustered data (just clouds & humidity  for now)")
        folder_list = [
            (pref + suff, *folder_params[pref])
            for pref in folders_pref
            for suff in folders_suff.values()
        ]

        for f, threshold, upper_lower, num_clusters in folder_list:
            #for the images
            resize_1_4_and_simplify(f"raw_data/extracted_fvg_cleaned/{f}", f"image_processing/fvg/resized/{f}")    
            generate_clustered_images(num_clusters, f"image_processing/fvg/resized/{f}", f"image_processing/fvg/clustered/{f}_clustered")
            pass    

        folder_list_clouds = [
            (folders_pref[0] + suff, folders_pref[0])
            for suff in folders_suff.values()
        ]

        #for the clouds run tobac   
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_tobac_merge_split, f"image_processing/fvg/clustered/{f}_clustered", f"image_processing/fvg/output_clustered/{f}", "raw_data/extracted_fvg_cleaned/borders.png", coordinates[2], coordinates[3], coordinates[0], coordinates[1], threshold, upper_lower,type_, n_min_threshold_clouds
                ):  (f,type_) for (f,type_)  in folder_list_clouds}

            for future in as_completed(futures):

                f = futures[future]
                try:
                    future.result()  # will raise exception if the call failed
                    print(f"✅ Completed {f}")
                except Exception as e:
                    print(f"❌ Error processing {f}: {e}")



        folder_list_humidity = [
            (folders_pref[1] + suff, folders_pref[1])
            for suff in folders_suff.values()
        ]

        #for the humidity   
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_tobac_fronts,f"image_processing/fvg/clustered/{f}_clustered", f"image_processing/fvg/output_clustered/{f}","raw_data/extracted_fvg_cleaned/borders.png",coordinates[2],coordinates[3],coordinates[0],coordinates[1],threshold,upper_lower,type_,n_min_threshold_fronts
                                ): (f,type_) for (f,type_) in folder_list_humidity}

            for future in as_completed(futures):
                
                f = futures[future]
                try:
                    future.result()  # will raise exception if the call failed
                    print(f"✅ Completed {f}")
                except Exception as e:
                    print(f"❌ Error processing {f}: {e}")

        #for the tmeperature
        folder_list_temp = [
            (folders_pref[2] + suff, folders_pref[2])
            for suff in folders_suff.values()
        ]
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_tobac_fronts,f"image_processing/fvg/clustered/{f}_clustered", f"image_processing/fvg/output_clustered/{f}","raw_data/extracted_fvg_cleaned/borders.png",coordinates[2],coordinates[3],coordinates[0],coordinates[1],threshold,upper_lower,type_,n_min_threshold_fronts
                                ): (f,type_) for (f,type_) in folder_list_temp}

            for future in as_completed(futures):
                
                f = futures[future]
                try:
                    future.result()  # will raise exception if the call failed
                    print(f"✅ Completed {f}")
                except Exception as e:
                    print(f"❌ Error processing {f}: {e}")


        base_path = "./image_processing/fvg/output_clustered/"
        generate_cloud_facts_over_cities(base_path)
        generate_humidity_facts_over_cities(base_path)

        starting_date=generate_temp_facts_over_cities(base_path)
        
        init_fronts_generation("./image_processing/fvg/output_clustered/", coordinates)
        generate_fronts_hum(starting_date)
        generate_fronts_temp(starting_date)



    elif mode == 3:
        #resize iamges once to work on smaller images and run TOBAC (FVG)
        print("run TOBAC on FVG data (just clouds & humidity for now)")
        folder_list = [
            (pref + suff, *folder_params[pref])
            for pref in folders_pref
            for suff in folders_suff.values()
        ]
        for f, threshold, upper_lower, _ in folder_list:
            resize_1_4_and_simplify(f"raw_data/extracted_fvg_cleaned/{f}", f"image_processing/fvg/resized/{f}")
        
        #for the temp
        folder_list_temp = [
            (folders_pref[2] + suff,  folders_pref[2])
            for suff in folders_suff.values()
        ]
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            print("threshold ", threshold )
            futures = {
                executor.submit(run_tobac_fronts,f"image_processing/fvg/resized/{f}", f"image_processing/fvg/output/{f}","raw_data/extracted_fvg_cleaned/borders.png",coordinates[2],coordinates[3],coordinates[0],coordinates[1],threshold,upper_lower,type_,n_min_threshold_fronts
                                ): (f,type_) for (f,type_) in folder_list_temp}

            for future in as_completed(futures):
                
                f = futures[future]
                try:
                    future.result()  # will raise exception if the call failed
                    print(f"✅ Completed {f}")
                except Exception as e:
                    print(f"❌ Error processing {f}: {e}")

        
        #for the clouds
        folder_list_clouds = [
            (folders_pref[0] + suff,  folders_pref[0])
            for suff in folders_suff.values()
        ]
        
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_tobac_merge_split, f"image_processing/fvg/resized/{f}", f"image_processing/fvg/output/{f}","raw_data/extracted_fvg_cleaned/borders.png", coordinates[2], coordinates[3], coordinates[0], coordinates[1], threshold, upper_lower,type_, n_min_threshold_fronts
                ): (f,type_) for (f,type_) in folder_list_clouds}

            for future in as_completed(futures):

                f = futures[future]
                try:
                    future.result()  # will raise exception if the call failed
                    print(f"✅ Completed {f}")
                except Exception as e:
                    print(f"❌ Error processing {f}: {e}")
        
        #for the humidity
        folder_list_humidity = [
            (folders_pref[1] + suff,  folders_pref[1])
            for suff in folders_suff.values()
        ]
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_tobac_fronts,f"image_processing/fvg/resized/{f}", f"image_processing/fvg/output/{f}","raw_data/extracted_fvg_cleaned/borders.png",coordinates[2],coordinates[3],coordinates[0],coordinates[1],threshold,upper_lower, type_, n_min_threshold_fronts
                                ): (f,type_) for (f,type_) in folder_list_humidity}

            for future in as_completed(futures):
                
                f = futures[future]
                try:
                    future.result()  # will raise exception if the call failed
                    print(f"✅ Completed {f}")
                except Exception as e:
                    print(f"❌ Error processing {f}: {e}")

      
        
        
        base_path = "./image_processing/fvg/output/"
        starting_date=generate_temp_facts_over_cities(base_path)

        init_fronts_generation("./image_processing/fvg/output/", coordinates)
        generate_fronts_hum(starting_date)
        generate_fronts_temp(starting_date)

    elif mode == 4:       

        generate_ground_truth()

    elif mode == 5:
        folder_list_clouds = [
            (folders_pref[0] + suff,  folders_pref[0])
            for suff in folders_suff.values()
        ]
        folder_list_humidity = [
            (folders_pref[1] + suff,  folders_pref[1])
            for suff in folders_suff.values()
        ]
        folder_list_temp = [
            (folders_pref[2] + suff, folders_pref[2])
            for suff in folders_suff.values()
        ]
        init_starting_date()
        merge_into_examples(folder_list_clouds,folder_list_humidity,folder_list_temp)
        
    elif mode == 6: #to experiment stuff
        base_path = "./image_processing/fvg/output/"
        starting_date=generate_temp_facts_over_cities(base_path)

        init_fronts_generation("./image_processing/fvg/output/", coordinates)
        generate_fronts_hum(starting_date)
        generate_fronts_temp(starting_date)
        

