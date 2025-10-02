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

    legend_dir = os.path.join(output_base, "legends")
    pathlib.Path(legend_dir).mkdir(parents=True, exist_ok=True)

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
        plt.savefig(os.path.join(legend_dir, f"legend_{lvl}hPa.png"), dpi=300, bbox_inches='tight')
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
