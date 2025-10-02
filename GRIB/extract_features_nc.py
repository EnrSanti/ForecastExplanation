import xarray as xr
import matplotlib
matplotlib.use("Agg")   

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import pandas as pd
import numpy as np
import pathlib

# ---- CONFIG ----

output_base = "./GRIB/extracted_fvg"

def save_feature_maps(input_path,coordinates):
    save_cloud_maps(input_path,coordinates)
    save_temperature_maps(input_path,coordinates)

def save_cloud_maps(input_path, coordinates):
    levels = [1000, 700, 500, 300]        
    folders = {1000: "cloud_at_100m", 700: "cloud_at_3km", 500: "cloud_at_5.5km", 300: "cloud_at_9km"}
    cmap = "Blues"

    ds = xr.open_dataset(input_path, decode_times=True, decode_timedelta=False)
    if 'ccl' not in ds:
        print("Error: 'ccl' variable not found in dataset.")
        return
    cloud = ds['ccl']

    # ---- CREATE OUTPUT FOLDERS ----
    for lvl in levels:
        pathlib.Path(os.path.join(output_base, folders[lvl])).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_base, "legends")).mkdir(parents=True, exist_ok=True)

    # ---- SAVE LEGEND PER LEVEL ----
    for lvl in levels:
        cloud_level = cloud.sel(isobaricInhPa=lvl)
        vmin = float(cloud_level.min())
        vmax = float(cloud_level.max())

        fig, ax = plt.subplots(figsize=(6,1))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax, orientation='horizontal'
        )
        cb.set_label(f'Cloud cover at {lvl} hPa [fraction]')
        plt.savefig(os.path.join(output_base, f"legend_{lvl}hPa_cloud.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ---- PLOT LOOP ----
    for lvl in levels:
        cloud_level = cloud.sel(isobaricInhPa=lvl)
        vmin = float(cloud_level.min())
        vmax = float(cloud_level.max())
        out_dir = os.path.join(output_base, folders[lvl])

        for i in range(cloud_level.sizes['time']):
            base_time = pd.to_datetime(str(cloud_level['time'].isel(time=i).values))
            for j in range(cloud_level.sizes['step']):
                step_val = int(cloud_level['step'].isel(step=j).values)
                valid_time = base_time + pd.Timedelta(hours=step_val)

                cloud_slice = cloud_level.isel(time=i, step=j)
                if not np.isfinite(cloud_slice).any():
                    continue

                fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent(coordinates, crs=ccrs.PlateCarree())
                pcm = ax.pcolormesh(
                    cloud_slice['longitude'], cloud_slice['latitude'], cloud_slice,
                    cmap=cmap, shading='auto', vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree()
                )
                ax.coastlines(resolution='10m', linewidth=1)
                ax.add_feature(cfeature.BORDERS, linewidth=0.8)
                ax.set_title(f"Cloud cover at {lvl} hPa\nValid time: {valid_time}")

                fname = os.path.join(out_dir, f"cloud_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png")
                plt.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)

    print("Finished plotting cloud maps with separate legends per level.")


def save_temperature_maps(input_path,coordinates):
    levels = [1000, 700, 500, 300]        # pressure levels
    folders = {1000: "temp_at_100m", 700: "temp_at_3km", 500: "temp_at_5.5km", 300: "temp_at_9km"}
    cmap = "coolwarm"

    # ---- OPEN DATASET ----
    ds = xr.open_dataset(input_path, decode_times=True, decode_timedelta=False)  # fix FutureWarning
    temperature = ds['t']

    # ---- CREATE OUTPUT FOLDERS ----
    for lvl in levels:
        folder_name = folders[lvl]
        out_dir = os.path.join(output_base, folder_name)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)


    # ---- SAVE LEGENDS ONCE PER LEVEL ----
    for lvl in levels:
        temp_level = temperature.sel(isobaricInhPa=lvl)
        vmin = float(temp_level.min())
        vmax = float(temp_level.max())

        fig, ax = plt.subplots(figsize=(6,1))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax, orientation='horizontal'
        )
        cb.set_label(f'Temperature at {lvl} hPa [K]')
        plt.savefig(os.path.join(output_base, f"legend_{lvl}hPa.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ---- PLOT LOOP ----
    for lvl in levels:
        temp_level = temperature.sel(isobaricInhPa=lvl)
        vmin = float(temp_level.min())
        vmax = float(temp_level.max())
        folder_name = folders[lvl]
        out_dir = os.path.join(output_base, folder_name)

        for i in range(temp_level.sizes['time']):
            base_time = pd.to_datetime(str(temp_level['time'].isel(time=i).values))

            for j in range(temp_level.sizes['step']):
                step_val = temp_level['step'].isel(step=j).values
                leadtime_hours = int(step_val)
                valid_time = base_time + pd.Timedelta(hours=leadtime_hours)

                temp_slice = temp_level.isel(time=i, step=j)
                if not np.isfinite(temp_slice).any():
                    continue

                fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent(coordinates, crs=ccrs.PlateCarree())

                pcm = ax.pcolormesh(
                    temp_slice['longitude'],
                    temp_slice['latitude'],
                    temp_slice,
                    cmap=cmap,
                    shading='auto',
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree()
                )
                ax.coastlines(resolution='10m', linewidth=1)
                ax.add_feature(cfeature.BORDERS, linewidth=0.8)
                ax.set_title(f"Temperature at {lvl} hPa\nValid time: {valid_time}")

                # <-- REMOVE colorbar here

                fname = os.path.join(out_dir, f"temp_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png")
                plt.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)


    print("Finished plotting all levels with consistent colormap and separate folders + legends.")

def print_nc_variables():
    ds = xr.open_dataset("./GRIB/data/2_9gb_cut.nc")
    
    # Unique variable names and their dimensions
    print("Variables and their dimensions:")
    for var in ds.data_vars:
        dims = ", ".join(ds[var].dims)
        print(f" - {var} ({dims})")
    
    # Unique coordinates (like levels, time, lat/lon)
    print("\nCoordinates and their sizes:")
    for coord in ds.coords:
        print(f" - {coord}: {ds.coords[coord].size}")

    # Optional: print levels if they exist
    if "level" in ds.coords:
        levels = ds.coords["level"].values
        print("\nUnique levels:")
        print(levels)

#print_nc_variables()