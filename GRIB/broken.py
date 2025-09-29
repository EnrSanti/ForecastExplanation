import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import concurrent.futures
import numpy as np
from concurrent.futures import ThreadPoolExecutor

grib_path = "./GRIB/data/ECMWF-ITA_2018010400.grib"

#for a single time step (and variable)
def plot_and_save(ds, coordinates, i, step, var):
    da = ds[var]

    # If the variable has a 'step' dimension, select the i-th step
    if 'step' in da.dims:
        da = da.isel(step=i)
        hours = int(step.values / np.timedelta64(1, "h"))
    else:
        hours = 0  # no time dimension

    lat = ds.latitude
    lon = ds.longitude

    # Ensure da is 2D
    if da.ndim != 2:
        raise ValueError(f"Variable {var} is not 2D after selecting step, shape={da.shape}")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    pcm = ax.pcolormesh(lon, lat, da, cmap='coolwarm', shading='nearest')

    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)

    if coordinates[0] is not None:
        ax.set_extent([coordinates[0], coordinates[1], coordinates[2], coordinates[3]], crs=ccrs.PlateCarree())

    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', label=f'{var}')

    ax.set_title(f"{var} map, step={hours}h")

    folder = "./GRIB/extracted_fvg" if coordinates[0] is not None else "./GRIB/extracted_it"
    plt.savefig(f"{folder}/{var}_map_step_{hours}h.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return hours



def process_variable(coordinates, var, level_type):
    global grib_path
    ds = xr.open_dataset(
        grib_path,
        engine='cfgrib',
        backend_kwargs={
            'filter_by_keys': {'shortName': var, 'typeOfLevel': level_type},
            'decode_times': True
        }
    )

    steps = ds.step if 'step' in ds.dims else [0]  # fallback for variables without step
    for i, step in enumerate(steps):
        plot_and_save(ds, coordinates, i, step, var)

    ds.close()
    print(f"Saved plots for {var}")


def save_feature_maps(coordinates, threads=4):
    
    global grib_path 
    variables_to_extract = {
    'msl': 'surface',
    't': 'isobaricInhPa',
    'u': 'isobaricInhPa',
    'v': 'isobaricInhPa',
    'swvl1': 'depthBelowLandLayer',
    'swvl3': 'depthBelowLandLayer',
    }
    # Run 1 thread per variable
    with ThreadPoolExecutor(max_workers=len(variables_to_extract)) as executor:
        futures = [executor.submit(process_variable,coordinates, var, level_type)
                for var, level_type in variables_to_extract.items()]
        for future in futures:
            future.result()
