import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import concurrent.futures
import numpy as np

import cfgrib
from cfgrib import open_from_index
# List all messages in GRIB
import pygrib

# GRIB (msl)
grib_path = "./GRIB/data/ECMWF-ITA_2018010100.grib"





 # For temperature (2m)

def save_feature_maps(coordinates,threads=4):

    
    save_msl_maps(coordinates, threads)
    save_temperature_maps(coordinates,[1000, 500], threads)

def plot_temperature(ds, coordinates, i, step, level_val, folderKm,unit='K'):
    # Extract temperature at selected level
    
    temp = ds.t.sel(isobaricInhPa=level_val).isel(step=i)
    lat = ds.latitude
    lon = ds.longitude

    # Convert timedelta to hours if step is timedelta
    if np.issubdtype(step.dtype, np.timedelta64):
        hours = int(step.values / np.timedelta64(1, "h"))
    else:
        hours = int(step)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    pcm = ax.pcolormesh(lon, lat, temp, cmap='coolwarm', shading='nearest')

    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)

    if coordinates[0] is not None:
        ax.set_extent([coordinates[0], coordinates[1], coordinates[2], coordinates[3]], crs=ccrs.PlateCarree())

    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', label=f'Temperature ({unit})')
    ax.set_title(f"Temperature at {level_val}, step={hours}h")

    folder = f"./GRIB/extracted_fvg/{folderKm}" if coordinates[0] is not None else f"./GRIB/extracted_it/{folderKm}"
    plt.savefig(f"{folder}/temp_{level_val}_step_{hours}h.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return hours

def save_temperature_maps(coordinates, levels, threads=4):
    ds = xr.open_dataset(
        grib_path,
        engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'shortName': 't'}, 'decode_timedelta': True}
    )

    steps = list(ds.step)
    folderKm = {
        1000: "temp_100m",  # or "2 m" if using that name
        500: "temp_5.5km"
    }
    for level in levels:
        folder = folderKm[level]  
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(plot_temperature, ds, coordinates, i, step, level,folder) 
                       for i, step in enumerate(steps)]
            for f in concurrent.futures.as_completed(futures):
                print(f"Finished step {f.result()}h at level {level}")
    ds.close()

def plot_msl(ds, coordinates, i, step):
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
        folder = "./GRIB/extracted_fvg/mean_sea_lv_pressure/"
    else: 
        folder = "./GRIB/extracted_it/mean_sea_lv_pressure/" 
    # Save figure
    plt.savefig( f"{folder}/msl_map_step_{hours}h.png", dpi=300, bbox_inches='tight' )
    plt.close(fig) 
    return hours


def save_msl_maps(coordinates, threads=4):
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
        futures = [executor.submit(plot_msl, ds,coordinates,i, step) for i, step in enumerate(steps)]
        for f in concurrent.futures.as_completed(futures):
            print(f"Finished step {f.result()}h")
    ds.close()


def print_grib_variables():
    grbs = pygrib.open('./GRIB/data/ECMWF-ITA_2018010100.grib')
    for grb in grbs:
        print(grb.name, grb.typeOfLevel, grb.level)

#print_grib_variables()