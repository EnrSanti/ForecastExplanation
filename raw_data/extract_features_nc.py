import gc
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.io import shapereader

from . import Region, CUT_DATA_DIR

logger = logging.getLogger(__name__)
matplotlib.use("Agg")

# define the hPa of the data considered, moreover define a more symbolic name for them
LEVELS = [1000, 925, 850, 700, 500, 300]
FOLDERS = {
    1000: "_at_100m",
    925: "_at_750m",
    850: "_at_1_4km",
    700: "_at_3km",
    500: "_at_5_5km",
    300: "_at_9km"
}

MINIMUM_CLOUD_VALUE = 0
MAXIMUM_CLOUD_VALUE = 100
MINIMUM_TEMP_VALUE = 215
MAXIMUM_TEMP_VALUE = 315
MINIMUM_WIND_SPEED_VALUE = 0
MAXIMUM_WIND_SPEED_VALUE = 100
MINIMUM_HUMIDITY_VALUE = 0
MAXIMUM_HUMIDITY_VALUE = 100


def extract_features_from_nc(dates: list, coordinates: Region, output_base: str):
    logger.info("Starting feature extraction from .nc files...")
    os.makedirs(output_base, exist_ok=True)

    save_borders_png(output_base, coordinates)
    create_legends(output_base)

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(save_feature_maps,
                            os.path.join(CUT_DATA_DIR, date.strftime("%Y-%m-%d") + "_" + coordinates.name + "_cut.nc"),
                            coordinates, output_base): date
            for date in dates
        }

        for future in as_completed(futures):
            date = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Feature extraction failed for {date}: {e}")

    logger.info("Feature extraction completed.")


def save_feature_maps(input_path, coordinates: Region, output_base: str):
    with xr.open_dataset(input_path, decode_times=True, decode_timedelta=False) as ds:
        if 'ccl' not in ds:
            print("Error: 'ccl' variable not found in dataset.")
            return

        if 'r' not in ds and 'rhum' not in ds:
            print("Error: 'r' (Relative Humidity) variable not found in dataset.")
            return

        save_cloud_maps(ds, coordinates, output_base)
        save_temperature_maps(ds, coordinates, output_base)
        save_wind_maps(ds, coordinates, output_base)
        save_humidity_maps(ds, coordinates, output_base)

        ds.close()

    logger.debug(f"Feature maps saved for {input_path} in {output_base}")


def save_borders_png(output_base, coordinates):
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())

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
        dpi=380,  # so no need to resize, we don't loose data in the borders
        bbox_inches="tight",
        pad_inches=0,
        transparent=True
    )
    plt.close(fig)


def create_legends(output_base):
    # Create legends for each level and feature
    # features = {
    #     "cloud": ("viridis", MINIMUM_CLOUD_VALUE, MAXIMUM_CLOUD_VALUE, "Cloud cover [%]"),
    #     "temp": ("OrRd", None, None, "Temperature [K]"),
    #     "winds": ("viridis", None, None, "Wind speed [m/s]"),
    #     "humidity": ("YlGnBu", None, None, "Relative humidity [%]")
    # }

    # CLOUD
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = plt.Normalize(vmin=MINIMUM_CLOUD_VALUE, vmax=MAXIMUM_CLOUD_VALUE)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        cax=ax,
        orientation='horizontal'
    )
    cb.set_label(f'Cloud cover [fraction]')
    legend_path = os.path.join(output_base, f"legend_cloud.png")
    plt.savefig(legend_path, dpi=130, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # TEMPERATURE
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = plt.Normalize(vmin=MINIMUM_TEMP_VALUE, vmax=MAXIMUM_TEMP_VALUE)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="OrRd"),
        cax=ax, orientation='horizontal'
    )
    cb.set_label(f'Temperature [K]')
    plt.savefig(os.path.join(output_base, f"legend_temp.png"), dpi=130, bbox_inches='tight',
                pad_inches=0)
    plt.close(fig)
    cmap_obj = plt.get_cmap("OrRd")
    rgba_min = cmap_obj(norm(MINIMUM_TEMP_VALUE))
    rgba_max = cmap_obj(norm(MAXIMUM_TEMP_VALUE))
    rgb255_min = tuple(int(255 * c) for c in rgba_min[:3])
    rgb255_max = tuple(int(255 * c) for c in rgba_max[:3])

    with open(os.path.join(output_base, f"legend_temp.txt"), 'w') as ftxt:
        ftxt.write(f"Temperature range: {MINIMUM_TEMP_VALUE:.2f} K to {MAXIMUM_TEMP_VALUE:.2f} K\n")
        ftxt.write(f"Respective colors: {rgb255_min}, {rgb255_max}\n")

    # WIND
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = plt.Normalize(vmin=MINIMUM_WIND_SPEED_VALUE, vmax=MAXIMUM_WIND_SPEED_VALUE)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        cax=ax,
        orientation="horizontal",
    )
    cb.set_label(f"Wind speed [m/s]")
    plt.savefig(
        os.path.join(output_base, f"legend_wind.png"),
        dpi=130,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    # HUMIDITY
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = plt.Normalize(vmin=MINIMUM_HUMIDITY_VALUE, vmax=MAXIMUM_HUMIDITY_VALUE)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="YlGnBu"),
        cax=ax, orientation='horizontal'
    )
    cb.set_label(f'Relative humidity [%]')
    plt.savefig(os.path.join(output_base, f"legend_humidity.png"),
                dpi=130, bbox_inches='tight')
    plt.close(fig)

    cmap_obj = plt.get_cmap("YlGnBu")
    rgba_min = cmap_obj(norm(MINIMUM_HUMIDITY_VALUE))
    rgba_max = cmap_obj(norm(MAXIMUM_HUMIDITY_VALUE))
    rgb255_min = tuple(int(255 * c) for c in rgba_min[:3])
    rgb255_max = tuple(int(255 * c) for c in rgba_max[:3])
    with open(os.path.join(output_base, f"legend_hum.txt"), 'w') as ftxt:
        ftxt.write(f"Humidity range: {MINIMUM_HUMIDITY_VALUE:.2f} K to {MAXIMUM_HUMIDITY_VALUE:.2f} K\n")
        ftxt.write(f"Respective colors: {rgb255_min}, {rgb255_max}\n")


def save_cloud_maps(ds, coordinates, output_base):
    """
       Given a xarray dataset, it saves the images with the data related to the clouds. Data at different height is stored in different folders

       Parameters
       ----------
       ds: the xarray dataset from which to extract the data
       coordinates: are the coordinates of the region to be plotted (extrema)
       output_base: the base directory where the extracted features will be saved
    """
    cloud_folders = {k: "cloud" + v for k, v in FOLDERS.items()}
    cloud = ds['ccl']

    # ---- PLOT LOOP ----
    for lvl in LEVELS:
        cloud_level = cloud.sel(isobaricInhPa=lvl)
        out_dir = os.path.join(output_base, cloud_folders[lvl])
        os.makedirs(out_dir, exist_ok=True)

        for i in range(cloud_level.sizes['time']):
            base_time = pd.to_datetime(str(cloud_level['time'].isel(time=i).values))

            # Allowed window: 01:00 of base day → 00:00 of next day
            day_start = base_time.normalize() + pd.Timedelta(hours=1)
            day_end = day_start + pd.Timedelta(days=1)

            for j in range(cloud_level.sizes['step']):
                step_val = int(cloud_level['step'].isel(step=j).values)
                valid_time = base_time + pd.Timedelta(hours=step_val)

                # ---- FILTER OUT OVERLAPPING FRAMES ----
                if not (day_start <= valid_time <= day_end):
                    continue

                cloud_slice = cloud_level.isel(time=i, step=j)
                if not np.isfinite(cloud_slice).any():
                    continue

                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())
                pcm = ax.pcolormesh(
                    cloud_slice['longitude'], cloud_slice['latitude'], cloud_slice,
                    cmap="viridis", shading='auto', vmin=MINIMUM_CLOUD_VALUE, vmax=MAXIMUM_CLOUD_VALUE,
                    transform=ccrs.PlateCarree()
                )
                fname = os.path.join(out_dir, f"cloud_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png")
                plt.savefig(fname, dpi=130, bbox_inches='tight', pad_inches=0)
                plt.close(fig)


def save_temperature_maps(ds, coordinates, output_base):
    """
       Given a xarray dataset, it saves the images with the data related to the temperatures. Data at different height is stored in different folders

       Parameters
       ----------
       ds: the xarray dataset from which to extract the data
       coordinates: are the coordinates of the region to be plotted (extrema)
       output_base: the base directory where the extracted features will be saved
    """
    temp_folders = {k: "temp" + v for k, v in FOLDERS.items()}
    temperature = ds['t']

    for lvl in LEVELS:
        temp_level = temperature.sel(isobaricInhPa=lvl)

        out_dir = os.path.join(output_base, temp_folders[lvl])
        os.makedirs(out_dir, exist_ok=True)

        for i in range(temp_level.sizes['time']):
            base_time = pd.to_datetime(str(temp_level['time'].isel(time=i).values))

            # Allowed window: 01:00 of base day → 00:00 of next day
            day_start = base_time.normalize() + pd.Timedelta(hours=1)
            day_end = day_start + pd.Timedelta(days=1)

            for j in range(temp_level.sizes['step']):
                step_val = temp_level['step'].isel(step=j).values
                leadtime_hours = int(step_val)
                valid_time = base_time + pd.Timedelta(hours=leadtime_hours)

                # ---- FILTER OUT OVERLAPS ----
                if not (day_start <= valid_time <= day_end):
                    continue

                temp_slice = temp_level.isel(time=i, step=j)
                if not np.isfinite(temp_slice).any():
                    continue

                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())

                pcm = ax.pcolormesh(
                    temp_slice['longitude'],
                    temp_slice['latitude'],
                    temp_slice,
                    cmap="OrRd",
                    shading='auto',
                    vmin=MINIMUM_TEMP_VALUE,
                    vmax=MAXIMUM_TEMP_VALUE,
                    transform=ccrs.PlateCarree()
                )

                fname = os.path.join(out_dir, f"temp_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png")
                plt.savefig(fname, dpi=130, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                gc.collect()


def save_wind_maps(ds, coordinates, output_base):
    """
       Given a xarray dataset, it saves the images with the data related to the wind (vectors included). Data at different height is stored in different folders

       Parameters
       ----------
        ds: the xarray dataset from which to extract the data
       coordinates: are the coordinates of the region to be plotted (extrema)
       output_base: the base directory where the extracted features will be saved
    """
    wind_folders = {k: "winds" + v for k, v in FOLDERS.items()}
    u_var = ds["u"]
    v_var = ds["v"]

    for lvl in LEVELS:
        u_lvl = u_var.sel(isobaricInhPa=lvl)
        v_lvl = v_var.sel(isobaricInhPa=lvl)

        out_dir = os.path.join(output_base, wind_folders[lvl])
        os.makedirs(out_dir, exist_ok=True)
        for i in range(u_lvl.sizes["time"]):
            base_time = pd.to_datetime(str(u_lvl["time"].isel(time=i).values))

            # Allowed window: 01:00 of base day → 00:00 of next day
            day_start = base_time.normalize() + pd.Timedelta(hours=1)
            day_end = day_start + pd.Timedelta(days=1)

            for j in range(u_lvl.sizes["step"]):
                step_val = int(u_lvl["step"].isel(step=j).values)
                valid_time = base_time + pd.Timedelta(hours=step_val)

                # ---- FILTER OUT OVERLAPS ----
                if not (day_start <= valid_time <= day_end):
                    continue

                u_slice = u_lvl.isel(time=i, step=j)
                v_slice = v_lvl.isel(time=i, step=j)
                wind_speed = np.sqrt(u_slice ** 2 + v_slice ** 2)

                if not np.isfinite(wind_speed).any():
                    continue

                # ---- FIGURE ----

                fig, ax = plt.subplots(
                    figsize=(10, 8),
                    dpi=380,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())

                pcm = ax.pcolormesh(
                    u_slice["longitude"],
                    u_slice["latitude"],
                    wind_speed,
                    cmap="viridis",
                    shading="auto",
                    vmin=MINIMUM_WIND_SPEED_VALUE,
                    vmax=MAXIMUM_WIND_SPEED_VALUE,
                    transform=ccrs.PlateCarree(),
                )

                # ---- SUBSAMPLE ----
                step = 10
                lon2d = u_slice["longitude"].broadcast_like(u_slice).values
                lat2d = u_slice["latitude"].broadcast_like(u_slice).values
                lon_subset = lon2d[::step, ::step]
                lat_subset = lat2d[::step, ::step]
                u_subset = u_slice.values[::step, ::step]
                v_subset = v_slice.values[::step, ::step]

                # ---- APPLY VISIBLE MASK ----
                lon_min, lon_max, lat_min, lat_max = coordinates.value
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

                txt_path = os.path.join(
                    output_base, f"wind_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.csv"
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
                fig.savefig(fname, dpi=130, bbox_inches="tight", pad_inches=0)
                plt.close(fig)


def save_humidity_maps(ds, coordinates, output_base):
    """
       Given a xarray dataset, it saves the images with the data related to the humidity. Data at different height is stored in different folders

       Parameters
       ----------
       ds: the xarray dataset from which to extract the data
       coordinates: are the coordinates of the region to be plotted (extrema)
       output_base: the base directory where the extracted features will be saved
    """

    humidity_folders = {k: "humidity" + v for k, v in FOLDERS.items()}
    humidity = ds['r'] if 'r' in ds else ds['rhum']

    # ---- PLOT LOOP ----
    for lvl in LEVELS:
        rh_level = humidity.sel(isobaricInhPa=lvl)
        out_dir = os.path.join(output_base, humidity_folders[lvl])
        os.makedirs(out_dir, exist_ok=True)

        for i in range(rh_level.sizes['time']):
            base_time = pd.to_datetime(str(rh_level['time'].isel(time=i).values))

            # Define allowed time window:
            # from 01:00 of base day to 00:00 of next day
            day_start = base_time.normalize() + pd.Timedelta(hours=1)
            day_end = day_start + pd.Timedelta(days=1)

            for j in range(rh_level.sizes['step']):
                step_val = int(rh_level['step'].isel(step=j).values)
                valid_time = base_time + pd.Timedelta(hours=step_val)

                # ---- FILTER ----
                if not (day_start <= valid_time <= day_end):
                    continue

                rh_slice = rh_level.isel(time=i, step=j)
                if not np.isfinite(rh_slice).any():
                    continue

                fig, ax = plt.subplots(figsize=(10, 8),
                                       subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())
                pcm = ax.pcolormesh(
                    rh_slice['longitude'], rh_slice['latitude'], rh_slice,
                    cmap="YlGnBu", shading='auto', vmin=MINIMUM_HUMIDITY_VALUE, vmax=MAXIMUM_HUMIDITY_VALUE,
                    transform=ccrs.PlateCarree()
                )

                fname = os.path.join(out_dir, f"humidity_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png")
                plt.savefig(fname, dpi=130, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            gc.collect()
