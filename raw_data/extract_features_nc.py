import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.io import shapereader

from . import Region, CUT_DATA_DIR
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

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


def extract_features_from_nc(dates: List[datetime], coordinates: Region, output_base: str) -> None:
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
                logger.error(f"Feature extraction failed for {date}", exc_info=True)

    logger.info("Feature extraction completed.")


def save_feature_maps(input_path: str, coordinates: Region, output_base: str) -> None:
    with xr.open_dataset(input_path) as ds:
        if any(var not in ds for var in ['ccl', 't', 'u', 'v', 'r']):
            logger.error("Error: One or more required variables not found in dataset.")
            return

        save_cloud_maps(ds, coordinates, output_base)
        save_temperature_maps(ds, coordinates, output_base)
        save_wind_maps(ds, coordinates, output_base)
        save_humidity_maps(ds, coordinates, output_base)

    logger.debug(f"Feature maps saved for {input_path} in {output_base}")


def save_borders_png(output_base: str, coordinates: Region) -> None:
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
        os.path.join(output_base, "borders.png"),
        dpi=130,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True
    )
    plt.close(fig)


def create_legends(output_base: str) -> None:
    features = {
        "cloud": {
            "cmap": "viridis",
            "vmin": MINIMUM_CLOUD_VALUE,
            "vmax": MAXIMUM_CLOUD_VALUE,
            "label": "Cloud cover [fraction]"
        },
        "temp": {
            "cmap": "OrRd",
            "vmin": MINIMUM_TEMP_VALUE,
            "vmax": MAXIMUM_TEMP_VALUE,
            "label": "Temperature [K]",
            "txt_prefix": "Temperature",
            "txt_unit": "K"
        },
        "wind": {
            "cmap": "viridis",
            "vmin": MINIMUM_WIND_SPEED_VALUE,
            "vmax": MAXIMUM_WIND_SPEED_VALUE,
            "label": "Wind speed [m/s]"
        },
        "humidity": {
            "cmap": "YlGnBu",
            "vmin": MINIMUM_HUMIDITY_VALUE,
            "vmax": MAXIMUM_HUMIDITY_VALUE,
            "label": "Relative humidity [%]",
            "txt_prefix": "Humidity",
            "txt_unit": "%"
        }
    }

    for key, props in features.items():
        fig, ax = plt.subplots(figsize=(6, 1))
        norm = plt.Normalize(vmin=props["vmin"], vmax=props["vmax"])

        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=props["cmap"]),
            cax=ax,
            orientation='horizontal'
        )
        cb.set_label(props["label"])

        png_path = os.path.join(output_base, f"legend_{key}.png")
        plt.savefig(png_path, dpi=130, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        if "txt_prefix" in props:
            cmap_obj = plt.get_cmap(props["cmap"])

            # Calculate RGB max/min
            rgba_min = cmap_obj(norm(props["vmin"]))
            rgba_max = cmap_obj(norm(props["vmax"]))
            rgb255_min = tuple(int(255 * c) for c in rgba_min[:3])
            rgb255_max = tuple(int(255 * c) for c in rgba_max[:3])

            txt_path = os.path.join(output_base, f"legend_{key}.txt")

            with open(txt_path, 'w') as ftxt:
                prefix = props["txt_prefix"]
                unit = props["txt_unit"]
                vmin, vmax = props["vmin"], props["vmax"]

                ftxt.write(f"{prefix} range: {vmin:.2f} {unit} to {vmax:.2f} {unit}\n")
                ftxt.write(f"Respective colors: {rgb255_min}, {rgb255_max}\n")


def save_cloud_maps(ds: xr.Dataset, coordinates: Region, output_base: str) -> None:
    cloud_folders = {k: "cloud" + v for k, v in FOLDERS.items()}
    cloud = ds['ccl']

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())
    
    for lvl in LEVELS:
        cloud_level = cloud.sel(isobaricInhPa=lvl)
        out_dir = os.path.join(output_base, cloud_folders[lvl])
        os.makedirs(out_dir, exist_ok=True)

        for i in range(cloud_level.sizes['time']):
            base_time = pd.to_datetime(str(cloud_level['time'].isel(time=i).values))
            day_start = base_time.normalize() + pd.Timedelta(hours=1)
            day_end = day_start + pd.Timedelta(days=1)

            for j in range(cloud_level.sizes['step']):
                step_val = int(cloud_level['step'].isel(step=j).values)
                valid_time = base_time + pd.Timedelta(nanoseconds=step_val)

                if not (day_start <= valid_time <= day_end):
                    continue

                cloud_slice = cloud_level.isel(time=i, step=j)
                if not np.isfinite(cloud_slice).any():
                    continue

                mesh = ax.pcolormesh(
                    cloud_slice['longitude'], cloud_slice['latitude'], cloud_slice,
                    cmap="viridis", shading='auto', vmin=MINIMUM_CLOUD_VALUE, vmax=MAXIMUM_CLOUD_VALUE,
                    transform=ccrs.PlateCarree()
                )
                fname = os.path.join(out_dir, f"cloud_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png")
                fig.savefig(fname, dpi=130, bbox_inches='tight', pad_inches=0)
                mesh.remove()
    plt.close(fig)


def save_temperature_maps(ds: xr.Dataset, coordinates: Region, output_base: str) -> None:
    temp_folders = {k: "temp" + v for k, v in FOLDERS.items()}
    temperature = ds['t']

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())
    
    for lvl in LEVELS:
        temp_level = temperature.sel(isobaricInhPa=lvl)
        out_dir = os.path.join(output_base, temp_folders[lvl])
        os.makedirs(out_dir, exist_ok=True)

        for i in range(temp_level.sizes['time']):
            base_time = pd.to_datetime(str(temp_level['time'].isel(time=i).values))
            day_start = base_time.normalize() + pd.Timedelta(hours=1)
            day_end = day_start + pd.Timedelta(days=1)

            for j in range(temp_level.sizes['step']):
                step_val = temp_level['step'].isel(step=j).values
                valid_time = base_time + pd.Timedelta(nanoseconds=int(step_val))

                if not (day_start <= valid_time <= day_end):
                    continue

                temp_slice = temp_level.isel(time=i, step=j)
                if not np.isfinite(temp_slice).any():
                    continue

                mesh = ax.pcolormesh(
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
                fig.savefig(fname, dpi=130, bbox_inches='tight', pad_inches=0)
                mesh.remove()
    plt.close(fig)


def save_wind_maps(ds: xr.Dataset, coordinates: Region, output_base: str) -> None:
    wind_folders = {k: "winds" + v for k, v in FOLDERS.items()}
    u_var = ds["u"]
    v_var = ds["v"]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=130, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())
    ax.axis("off")
    
    for lvl in LEVELS:
        u_lvl = u_var.sel(isobaricInhPa=lvl)
        v_lvl = v_var.sel(isobaricInhPa=lvl)

        out_dir = os.path.join(output_base, wind_folders[lvl])
        os.makedirs(out_dir, exist_ok=True)
        for i in range(u_lvl.sizes["time"]):
            base_time = pd.to_datetime(str(u_lvl["time"].isel(time=i).values))
            day_start = base_time.normalize() + pd.Timedelta(hours=1)
            day_end = day_start + pd.Timedelta(days=1)

            for j in range(u_lvl.sizes["step"]):
                step_val = int(u_lvl["step"].isel(step=j).values)
                valid_time = base_time + pd.Timedelta(nanoseconds=step_val)

                if not (day_start <= valid_time <= day_end):
                    continue

                u_slice = u_lvl.isel(time=i, step=j)
                v_slice = v_lvl.isel(time=i, step=j)
                wind_speed = np.sqrt(u_slice ** 2 + v_slice ** 2)

                if not np.isfinite(wind_speed).any():
                    continue

                mesh = ax.pcolormesh(
                    u_slice["longitude"],
                    u_slice["latitude"],
                    wind_speed,
                    cmap="viridis",
                    shading="auto",
                    vmin=MINIMUM_WIND_SPEED_VALUE,
                    vmax=MAXIMUM_WIND_SPEED_VALUE,
                    transform=ccrs.PlateCarree(),
                )

                step = 10
                lon2d = u_slice["longitude"].broadcast_like(u_slice).values
                lat2d = u_slice["latitude"].broadcast_like(u_slice).values
                lon_subset = lon2d[::step, ::step]
                lat_subset = lat2d[::step, ::step]
                u_subset = u_slice.values[::step, ::step]
                v_subset = v_slice.values[::step, ::step]

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

                fig.canvas.draw()
                bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                x0, y0, width_in, height_in = bbox.x0, bbox.y0, bbox.width, bbox.height

                disp = ax.transData.transform(np.column_stack((lon_visible, lat_visible)))

                px = np.round(disp[:, 0] - x0 * fig.dpi).astype(int)
                py = np.round(int(height_in * fig.dpi) - (disp[:, 1] - y0 * fig.dpi)).astype(int)

                mags = np.sqrt(u_visible ** 2 + v_visible ** 2)
                alphas = np.degrees(np.arctan2(v_visible, u_visible))
                alphas[~np.isfinite(alphas)] = 0
                mags[~np.isfinite(mags)] = 0

                txt_path = os.path.join(out_dir, f"wind_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.csv")
                with open(txt_path, "w") as ftxt:
                    ftxt.write("vector_id,pixel_x,pixel_y,latitude,longitude,magnitude,alpha_deg")
                    for idx in range(len(lon_visible)):
                        ftxt.write(f"{idx},{px[idx]},{py[idx]},{lat_visible[idx]:.6f},{lon_visible[idx]:.6f},{mags[idx]:.6f},{alphas[idx]:.2f}")

                fname = os.path.join(out_dir, f"wind_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png")
                fig.savefig(fname, dpi=130, bbox_inches="tight", pad_inches=0)
                mesh.remove()
    plt.close(fig)


def save_humidity_maps(ds: xr.Dataset, coordinates: Region, output_base: str) -> None:
    humidity_folders = {k: "humidity" + v for k, v in FOLDERS.items()}
    humidity = ds['r'] if 'r' in ds else ds['rhum']

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())
    
    for lvl in LEVELS:
        rh_level = humidity.sel(isobaricInhPa=lvl)
        out_dir = os.path.join(output_base, humidity_folders[lvl])
        os.makedirs(out_dir, exist_ok=True)

        for i in range(rh_level.sizes['time']):
            base_time = pd.to_datetime(str(rh_level['time'].isel(time=i).values))
            day_start = base_time.normalize() + pd.Timedelta(hours=1)
            day_end = day_start + pd.Timedelta(days=1)

            for j in range(rh_level.sizes['step']):
                step_val = int(rh_level['step'].isel(step=j).values)
                valid_time = base_time + pd.Timedelta(nanoseconds=step_val)

                if not (day_start <= valid_time <= day_end):
                    continue

                rh_slice = rh_level.isel(time=i, step=j)
                if not np.isfinite(rh_slice).any():
                    continue

                mesh = ax.pcolormesh(
                    rh_slice['longitude'], rh_slice['latitude'], rh_slice,
                    cmap="YlGnBu", shading='auto', vmin=MINIMUM_HUMIDITY_VALUE, vmax=MAXIMUM_HUMIDITY_VALUE,
                    transform=ccrs.PlateCarree()
                )

                fname = os.path.join(out_dir, f"humidity_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png")
                fig.savefig(fname, dpi=130, bbox_inches='tight', pad_inches=0)
                mesh.remove()
    plt.close(fig)
