import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import concurrent.futures
import numpy as np

# GRIB (msl)
grib_path = "./GRIB/data/ECMWF-ITA_2018010100.grib"


#for a single time step
def plot_and_save_italy(ds,coordinates,i, step):
    msl = ds.msl.isel(step=i)
    lat = ds.latitude
    lon = ds.longitude

    # Convert timedelta to hours
    hours = int(step.values / np.timedelta64(1, "h"))

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    pcm = ax.pcolormesh(lon, lat, msl, cmap='coolwarm', shading='nearest')

    # Add coastlines, borders, and gridlines
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)

    if(coordinates[0] is not None):
        ax.set_extent([coordinates[0], coordinates[1], coordinates[2], coordinates[3]], crs=ccrs.PlateCarree()) 
    # Add colorbar
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', label='Mean Sea Level Pressure (Pa)')
    # Title
    ax.set_title(f"MSL Pressure, step={hours}h") 
    if(coordinates[0] is not None):
        folder = "./GRIB/extracted_fvg"
    else: 
        folder = "./GRIB/extracted_it" 
    # Save figure
    plt.savefig( f"{folder}/msl_map_step_{hours}h.png", dpi=300, bbox_inches='tight' )
    plt.close(fig) 
    return hours

def save_feature_maps(coordinates,threads=4):

    ds = xr.open_dataset(
        grib_path,
        engine='cfgrib',
        backend_kwargs={
            'filter_by_keys': {'shortName': 'msl', 'typeOfLevel': 'surface'},
            'decode_timedelta': True
        }
    )
    #for each t_step in the GRIB one thread (up to 4) saves the feature map  
    steps = list(ds.step)
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(plot_and_save_italy, ds,coordinates,i, step) for i, step in enumerate(steps)]
        for f in concurrent.futures.as_completed(futures):
            print(f"Finished step {f.result()}h")
    ds.close()



 