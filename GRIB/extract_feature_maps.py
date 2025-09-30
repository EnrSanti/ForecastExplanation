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
grib_path = "./GRIB/data/ECMWF-ITA_2018010200.grib"



def save_colorbar(vmin, vmax, cmap, label, outpath):
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, orientation='horizontal'
    )
    cb.set_label(label)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)


 # For temperature (2m)

def save_feature_maps(coordinates,threads=4):

    
    save_msl_maps(coordinates, threads)
    save_temperature_maps(coordinates, [1000, 700, 500, 400], threads)


def plot_temperature(ds, coordinates, i, step, level_val, folderKm, vmin, vmax, unit='K'):
    temp = ds.t.sel(isobaricInhPa=level_val).isel(step=i)
    lat = ds.latitude
    lon = ds.longitude

    if np.issubdtype(step.dtype, np.timedelta64):
        hours = int(step.values / np.timedelta64(1, "h"))
    else:
        hours = int(step)

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    pcm = ax.pcolormesh(lon, lat, temp, cmap='coolwarm', shading='nearest',
                        vmin=vmin, vmax=vmax)

    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)

    if coordinates[0] is not None:
        ax.set_extent([coordinates[0], coordinates[1], coordinates[2], coordinates[3]], crs=ccrs.PlateCarree())

    ax.set_title(f"Temperature at {level_val} hPa, step={hours}h")

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
        1000: "temp_100m",
        700:  "temp_3km",
        500:  "temp_5.5km",
        400:  "temp_7km"
    }

    for level in levels:
        folder = folderKm[level]

        # compute global min/max for colormap
        vmin = float(ds.t.sel(isobaricInhPa=level).min())
        vmax = float(ds.t.sel(isobaricInhPa=level).max())

        # save separate legend
        legend_file = f"./GRIB/extracted_fvg/{folder}_legend.png" if coordinates[0] is not None else f"./GRIB/extracted_it/{folder}_legend.png"
        save_colorbar(vmin, vmax, cmap='coolwarm', label=f"Temperature (K) at {level} hPa", outpath=legend_file)

        # parallel plots
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(plot_temperature, ds, coordinates, i, step, level, folder, vmin, vmax)
                       for i, step in enumerate(steps)]
            for f in concurrent.futures.as_completed(futures):
                print(f"Finished step {f.result()}h at level {level}")

    ds.close()


def plot_msl(ds, coordinates, i, step, vmin, vmax):
    msl = ds.msl.isel(step=i)
    lat = ds.latitude
    lon = ds.longitude

    hours = int(step.values / np.timedelta64(1, "h"))

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    pcm = ax.pcolormesh(lon, lat, msl, cmap='coolwarm', shading='nearest',
                        vmin=vmin, vmax=vmax)

    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)

    if coordinates[0] is not None:
        ax.set_extent([coordinates[0], coordinates[1], coordinates[2], coordinates[3]], crs=ccrs.PlateCarree())

    ax.set_title(f"MSL Pressure, step={hours}h")

    if coordinates[0] is not None:
        folder = "./GRIB/extracted_fvg/mean_sea_lv_pressure/"
    else:
        folder = "./GRIB/extracted_it/mean_sea_lv_pressure/"

    plt.savefig(f"{folder}/msl_map_step_{hours}h.png", dpi=300, bbox_inches='tight')
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

    steps = list(ds.step)

    # compute global min/max for MSL across all timesteps
    vmin = float(ds.msl.min())
    vmax = float(ds.msl.max())

    # choose folder based on coordinates
    if coordinates[0] is not None:
        folder = "./GRIB/extracted_fvg/mean_sea_lv_pressure/"
    else:
        folder = "./GRIB/extracted_it/mean_sea_lv_pressure/"

    # save standalone colorbar
    legend_file = f"./GRIB/extracted_fvg/msl_legend.png" if coordinates[0] is not None else f"./GRIB/extracted_it/msl_legend.png"
    save_colorbar(vmin, vmax, cmap='coolwarm', label="Mean Sea Level Pressure (Pa)", outpath=legend_file)

    # parallel plotting
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(plot_msl, ds, coordinates, i, step, vmin, vmax) for i, step in enumerate(steps)]
        for f in concurrent.futures.as_completed(futures):
            print(f"Finished step {f.result()}h")

    ds.close()


def print_grib_variables():
    grbs = pygrib.open('./GRIB/data/ECMWF-ITA_2018010200.grib')
    for grb in grbs:
        print(grb.name, grb.typeOfLevel, grb.level)

#print_grib_variables()