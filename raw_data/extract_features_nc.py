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
from cartopy.io import shapereader

#define the hPa of the data considered, moreover define a more symbolic name for them
levels = [1000, 925, 850, 700, 500, 300]
folders= {
    1000: "_at_100m", 
    925: "_at_750m",
    850: "_at_1_4km",
    700: "_at_3km", 
    500: "_at_5_5km", 
    300: "_at_9km"
}
output_base = ""

def save_feature_maps(input_path,coordinates, is_fvg, clean_plot):
    """
       Given a .nc file path, it saves the images with the data related to the temperature,humidity,wind and clouds at all levels.

       Parameters
       ----------
       input_path: the file from which to extract the data
       coordinates: [longmin, longmax, latmin, latmax], are the coordinates (extrema) of the image plot)
       is_fvg: boolean field used to store the extracted data in the prorper folder (true -> the folder will have suffix "fvg" otherwise "it")
       clean_plot: true -> in the image no political borders of the region are plotted, false -> otherwise.       
    """
    global output_base
    if(is_fvg):
        output_base = "./raw_data/extracted_fvg"
    else:
        output_base = "./raw_data/extracted_it"
        
    if(clean_plot):
        output_base = output_base+"_cleaned"
    

    
    print(f"Output base directory: {output_base}")
    save_borders_png(output_base,coordinates)
    save_humidity_maps(input_path, coordinates,clean_plot)
    save_cloud_maps(input_path,coordinates, clean_plot)
    save_wind_maps(input_path, coordinates,clean_plot)
    save_temperature_maps(input_path,coordinates, clean_plot)




def save_borders_png(output_base, coordinates):
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(coordinates, crs=ccrs.PlateCarree())

    # --- National borders and coastlines ---
    ax.coastlines(resolution='10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='black')

    # --- Regional borders (Italy only) ---
    shpfilename = shapereader.natural_earth(
        resolution='10m',
        category='cultural',
        name='admin_1_states_provinces'
    )
    reader = shapereader.Reader(shpfilename)
    for record in reader.records():
        if record.attributes.get("adm0_a3") == "ITA":  # Only Italy
            geom = record.geometry
            ax.add_geometries([geom],
                              crs=ccrs.PlateCarree(),
                              facecolor='none',
                              edgecolor='gray',
                              linewidth=0.6,
                              linestyle='--')

    # --- Styling and export ---
    ax.axis("off")
    fig.savefig(
        output_base + "/borders.png",
        dpi=380, #so no need to resize, we don't loose data in the borders
        bbox_inches="tight",
        pad_inches=0,
        transparent=True
    )
    plt.close(fig)


def save_cloud_maps(input_path, coordinates,clean_plot):
    """
       Given a .nc file path, it saves the images with the data related to the clouds. Data at different height is stored in different folders 

       Parameters
       ----------
       input_path: the file from which to extract the data
       coordinates: [longmin, longmax, latmin, latmax], are the coordinates (extrema) of the image plot)
       clean_plot: true -> in the image no political borders of the region are plotted, false -> otherwise.       
    """
    global levels,folders 
    
    cloud_folders = {k: "cloud" + v for k, v in folders.items()}


    cmap = "Blues"

    ds = xr.open_dataset(input_path, decode_times=True, decode_timedelta=False)
    if 'ccl' not in ds:
        print("Error: 'ccl' variable not found in dataset.")
        return
    cloud = ds['ccl']

    # ---- CREATE OUTPUT FOLDERS ----
    for lvl in levels:
        pathlib.Path(os.path.join(output_base, cloud_folders[lvl])).mkdir(parents=True, exist_ok=True)

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
        plt.savefig(os.path.join(output_base, f"legend{folders[lvl]}_cloud.png"), dpi=380, bbox_inches='tight')
        plt.close(fig)

    # ---- PLOT LOOP ----
    for lvl in levels:
        cloud_level = cloud.sel(isobaricInhPa=lvl)
        vmin = float(cloud_level.min())
        vmax = float(cloud_level.max())
        out_dir = os.path.join(output_base, cloud_folders[lvl])

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
                if not clean_plot:
                    ax.coastlines(resolution='10m', linewidth=1)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.8)
                    #ax.set_title(f"Cloud cover at {lvl} hPa\nValid time: {valid_time}")

                fname = os.path.join(out_dir, f"cloud_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png")
                plt.savefig(fname, dpi=380, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

    print("Finished plotting cloud maps with separate legends per level.")

def save_temperature_maps(input_path,coordinates, clean_plot):
    """
       Given a .nc file path, it saves the images with the data related to the temperatures. Data at different height is stored in different folders 

       Parameters
       ----------
       input_path: the file from which to extract the data
       coordinates: [longmin, longmax, latmin, latmax], are the coordinates (extrema) of the image plot)
       clean_plot: true -> in the image no political borders of the region are plotted, false -> otherwise.       
    """
    global levels,folders
    temp_folders = {k: "temp" + v for k, v in folders.items()}

    cmap = "coolwarm"

    # ---- OPEN DATASET ----
    ds = xr.open_dataset(input_path, decode_times=True, decode_timedelta=False)  # fix FutureWarning
    temperature = ds['t']

    # ---- CREATE OUTPUT FOLDERS ----
    for lvl in levels:
        folder_name = temp_folders[lvl]
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
        plt.savefig(os.path.join(output_base, f"legend{folders[lvl]}_temp.png"), dpi=380, bbox_inches='tight',pad_inches=0)
        plt.close(fig)

    # ---- PLOT LOOP ----
    for lvl in levels:
        temp_level = temperature.sel(isobaricInhPa=lvl)
        vmin = float(temp_level.min())
        vmax = float(temp_level.max())
        folder_name = temp_folders[lvl]
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
                if not clean_plot:
                    ax.coastlines(resolution='10m', linewidth=1)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.8)
                    #ax.set_title(f"Temperature at {lvl} hPa\nValid time: {valid_time}")


                fname = os.path.join(out_dir, f"temp_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png")
                plt.savefig(fname, dpi=380, bbox_inches='tight',pad_inches=0)
                plt.close(fig)


    print("Finished plotting all levels with consistent colormap and separate folders + legends.")

def save_wind_maps(input_path, coordinates, clean_plot):
    """
       Given a .nc file path, it saves the images with the data related to the wind (vectors included). Data at different height is stored in different folders 

       Parameters
       ----------
       input_path: the file from which to extract the data
       coordinates: [longmin, longmax, latmin, latmax], are the coordinates (extrema) of the image plot)
       clean_plot: true -> in the image no political borders of the region are plotted, false -> otherwise.       
    """

    global levels,folders
    wind_folders = {k: "winds" + v for k, v in folders.items()}
    cmap = "viridis"

    ds = xr.open_dataset(input_path, decode_times=True, decode_timedelta=False)
    u_var = ds["u"]
    v_var = ds["v"]

    # ---- CREATE OUTPUT FOLDERS ----
    for lvl in levels:
        pathlib.Path(os.path.join(output_base, wind_folders[lvl])).mkdir(parents=True, exist_ok=True)

    for lvl in levels:
        u_lvl = u_var.sel(isobaricInhPa=lvl)
        v_lvl = v_var.sel(isobaricInhPa=lvl)

        # Compute wind speed range
        wind_speed = np.sqrt(u_lvl ** 2 + v_lvl ** 2)
        vmin = float(np.nanmin(wind_speed))
        vmax = float(np.nanmax(wind_speed))

        # ---- SAVE LEGEND PER LEVEL ----
        fig, ax = plt.subplots(figsize=(6, 1))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
        )
        cb.set_label(f"Wind speed at {lvl} hPa [m/s]")
        plt.savefig(
            os.path.join(output_base, f"legend{folders[lvl]}_wind.png"),
            dpi=380,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

        out_dir = os.path.join(output_base, wind_folders[lvl])
    
    for lvl in levels:
        for i in range(u_lvl.sizes["time"]):
            base_time = pd.to_datetime(str(u_lvl["time"].isel(time=i).values))
            for j in range(u_lvl.sizes["step"]):
                step_val = int(u_lvl["step"].isel(step=j).values)
                valid_time = base_time + pd.Timedelta(hours=step_val)

                u_slice = u_lvl.isel(time=i, step=j)
                v_slice = v_lvl.isel(time=i, step=j)
                wind_speed = np.sqrt(u_slice ** 2 + v_slice ** 2)

                if not np.isfinite(wind_speed).any():
                    continue

                # ---- FIGURE ----
                
                fig, ax = plt.subplots(
                    figsize=(10, 8),
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                ax.set_extent(coordinates, crs=ccrs.PlateCarree())

                pcm = ax.pcolormesh(
                    u_slice["longitude"],
                    u_slice["latitude"],
                    wind_speed,
                    cmap=cmap,
                    shading="auto",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                )

                if not clean_plot:
                    ax.coastlines(resolution="10m", linewidth=1)
                    ax.add_feature(cfeature.BORDERS, linestyle=":")

                # ---- SUBSAMPLE ----
                step = 10
                lon2d = u_slice["longitude"].broadcast_like(u_slice).values
                lat2d = u_slice["latitude"].broadcast_like(u_slice).values
                lon_subset = lon2d[::step, ::step]
                lat_subset = lat2d[::step, ::step]
                u_subset = u_slice.values[::step, ::step]
                v_subset = v_slice.values[::step, ::step]

                # ---- APPLY VISIBLE MASK ----
                lon_min, lon_max, lat_min, lat_max = coordinates
                mask = (
                    (lon_subset >= lon_min)
                    & (lon_subset <= lon_max)
                    & (lat_subset >= lat_min)
                    & (lat_subset <= lat_max)
                )

                lon_visible = lon_subset[mask]
                lat_visible = lat_subset[mask]
                u_visible = u_subset[mask]
                v_visible = v_subset[mask]

                # ---- COMPUTE PIXEL COORDS (axes-relative, DPI-invariant) ----
                fig.canvas.draw()
                bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                x0, y0, width_in, height_in = bbox.x0, bbox.y0, bbox.width, bbox.height
                width_px = int(width_in * fig.dpi)
                height_px = int(height_in * fig.dpi)

                disp = ax.transData.transform(np.column_stack((lon_visible, lat_visible)))

                px = np.round(disp[:, 0] - x0 * fig.dpi).astype(int)
                py = np.round(height_px - (disp[:, 1] - y0 * fig.dpi)).astype(int)

                # ---- MAGNITUDE & ANGLE ----
                mags = np.sqrt(u_visible ** 2 + v_visible ** 2)
                alphas = np.degrees(np.arctan2(v_visible, u_visible))
                alphas[~np.isfinite(alphas)] = 0
                mags[~np.isfinite(mags)] = 0

                # ---- CLEAN VS NON-CLEAN ----
                if clean_plot:
                    txt_path = os.path.join(
                        out_dir, f"wind_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.txt"
                    )
                    with open(txt_path, "w") as ftxt:
                        ftxt.write(
                            "vector_id,pixel_x,pixel_y,latitude,longitude,magnitude,alpha_deg\n"
                        )
                        for idx in range(len(lon_visible)):
                            ftxt.write(
                                f"{idx},{px[idx]},{py[idx]},{lat_visible[idx]:.6f},{lon_visible[idx]:.6f},{mags[idx]:.6f},{alphas[idx]:.2f}\n"
                            )
                else:
                    # Draw arrows
                    ax.quiver(
                        lon_subset,
                        lat_subset,
                        u_subset,
                        v_subset,
                        color="black",
                        width=0.0015,
                        pivot="middle",
                        alpha=0.8,
                        scale=800,
                        transform=ccrs.PlateCarree(),
                    )
                    # Annotate visible ones
                    txt_path = os.path.join(
                        out_dir, f"wind_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.txt"
                    )
                    with open(txt_path, "w") as ftxt:
                        ftxt.write(
                            "vector_id,pixel_x,pixel_y,latitude,longitude,magnitude,alpha_deg\n"
                        )
                        for idx in range(len(lon_visible)):
                            ftxt.write(
                                f"{idx},{px[idx]},{py[idx]},{lat_visible[idx]:.6f},{lon_visible[idx]:.6f},{mags[idx]:.6f},{alphas[idx]:.2f}\n"
                            )

                # ---- SAVE FIGURE ----
                ax.axis("off")
                fname = os.path.join(
                    out_dir, f"wind_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png"
                )
                fig.savefig(fname, dpi=380, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

    ds.close()
    print("Finished plotting wind maps (aligned pixel coordinates).")

def save_humidity_maps(input_path, coordinates, clean_plot):

    """
       Given a .nc file path, it saves the images with the data related to the humidity. Data at different height is stored in different folders 

       Parameters
       ----------
       input_path: the file from which to extract the data
       coordinates: [longmin, longmax, latmin, latmax], are the coordinates (extrema) of the image plot)
       clean_plot: true -> in the image no political borders of the region are plotted, false -> otherwise.       
    """

    global levels, folders 
    
    humidity_folders = {k: "humidity" + v for k, v in folders.items()}
    cmap = "YlGnBu"

    ds = xr.open_dataset(input_path, decode_times=True, decode_timedelta=False)
    if 'r' not in ds and 'rhum' not in ds:
        print("Error: 'r' (Relative Humidity) variable not found in dataset.")
        return
    humidity = ds['r'] if 'r' in ds else ds['rhum']

    # ---- CREATE OUTPUT FOLDERS ----
    for lvl in levels:
        pathlib.Path(os.path.join(output_base, humidity_folders[lvl])).mkdir(parents=True, exist_ok=True)

    # ---- SAVE LEGEND PER LEVEL ----
    for lvl in levels:
        rh_level = humidity.sel(isobaricInhPa=lvl)
        vmin = float(rh_level.min())
        vmax = float(rh_level.max())

        fig, ax = plt.subplots(figsize=(6,1))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax, orientation='horizontal'
        )
        cb.set_label(f'Relative humidity at {lvl} hPa [%]')
        plt.savefig(os.path.join(output_base, f"legend_at{folders[lvl]}_humidity.png"),
                    dpi=380, bbox_inches='tight')
        plt.close(fig)

    # ---- PLOT LOOP ----
    for lvl in levels:
        rh_level = humidity.sel(isobaricInhPa=lvl)
        vmin = float(rh_level.min())
        vmax = float(rh_level.max())
        out_dir = os.path.join(output_base, humidity_folders[lvl])

        for i in range(rh_level.sizes['time']):
            base_time = pd.to_datetime(str(rh_level['time'].isel(time=i).values))
            for j in range(rh_level.sizes['step']):
                step_val = int(rh_level['step'].isel(step=j).values)
                valid_time = base_time + pd.Timedelta(hours=step_val)

                rh_slice = rh_level.isel(time=i, step=j)
                if not np.isfinite(rh_slice).any():
                    continue

                fig, ax = plt.subplots(figsize=(10,8),
                                       subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent(coordinates, crs=ccrs.PlateCarree())
                pcm = ax.pcolormesh(
                    rh_slice['longitude'], rh_slice['latitude'], rh_slice,
                    cmap=cmap, shading='auto', vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree()
                )
                if not clean_plot:
                    ax.coastlines(resolution='10m', linewidth=1)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.8)
                    #ax.set_title(f"Relative humidity at {lvl} hPa\nValid time: {valid_time}")

                fname = os.path.join(out_dir, f"humidity_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png")
                plt.savefig(fname, dpi=380, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

    print("Finished plotting humidity maps with separate legends per level.")
