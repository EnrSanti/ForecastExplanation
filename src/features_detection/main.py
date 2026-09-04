import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional

import matplotlib
import pandas as pd
import xarray as xr
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from features_detection.constants import (
    DEFAULT_GAP_FRAMES,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_SMOOTH,
    DEFAULT_V_MAX_AT_HEIGHT,
    FOLDERS_HEIGHT_SUFF,
    WeatherPhenomenon,
    WeatherPhenomenonTobacParams,
)
from features_detection.features import (
    detect_features,
    segment_features,
    track_features,
)
from features_detection.utils import (
    build_referenced_data_from_xarray,
    get_grid_spacings,
    normalize_referenced_data,
)
from region import Region

logger = logging.getLogger(__name__)


def run_tobac(
    dates: List[datetime],
    input_dir: str,
    output_dir: str,
    region: Region,
    force: bool = False,
    save_images: bool = False,
):
    """
    Executes TOBAC tracking across the specified list of dates and weather phenomena.
    """
    logger.info(f"Starting TOBAC.")
    os.makedirs(output_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(
                _run_tobac_single_day,
                date,
                input_dir,
                output_dir,
                region,
                force=force,
                save_images=save_images,
            ): date
            for date in dates
        }

        for future in tqdm(
            as_completed(futures), total=len(dates), desc="TOBAC Processing"
        ):
            date = futures[future]
            try:
                future.result()
            except Exception:
                logger.error(f"TOBAC failed for {date}", exc_info=True)

    logger.info("TOBAC runs completed.")


def _run_tobac_single_day(
    date: datetime,
    input_dir: str,
    output_dir: str,
    region: Region,
    force: bool = False,
    save_images: bool = False,
):
    day_input_dir = os.path.join(input_dir, date.strftime("%Y-%m-%d"))
    day_output_dir = os.path.join(output_dir, date.strftime("%Y-%m-%d"))
    os.makedirs(day_output_dir, exist_ok=True)

    if not force and os.path.exists(os.path.join(day_output_dir, "segmentation.nc")):
        logger.debug(
            f"Segmentation already exists for {date.strftime('%Y-%m-%d')}. Skipping."
        )
        return

    temp_tra_df, temp_seg_ds = _run_tobac_single_day_single_phenomenon(
        day_input_dir,
        day_output_dir,
        region,
        WeatherPhenomenon.TEMPERATURE,
        WeatherPhenomenonTobacParams.TEMPERATURE,
        save_images,
    )
    hum_tra_df, hum_seg_ds = _run_tobac_single_day_single_phenomenon(
        day_input_dir,
        day_output_dir,
        region,
        WeatherPhenomenon.HUMIDITY,
        WeatherPhenomenonTobacParams.HUMIDITY,
        save_images,
    )
    cld_tra_df, cld_seg_ds = _run_tobac_single_day_single_phenomenon(
        day_input_dir,
        day_output_dir,
        region,
        WeatherPhenomenon.CLOUDS,
        WeatherPhenomenonTobacParams.CLOUDS,
        save_images,
    )

    wind_seg_ds = _extract_winds(day_input_dir)

    dfs = [df for df in [temp_tra_df, hum_tra_df, cld_tra_df] if not df.empty]
    results_tra = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    del temp_tra_df, hum_tra_df, cld_tra_df

    dss = [
        ds for ds in [temp_seg_ds, hum_seg_ds, cld_seg_ds, wind_seg_ds] if ds.data_vars
    ]
    results_seg_ds = xr.merge(dss, compat="override", join="outer")
    del temp_seg_ds, hum_seg_ds, cld_seg_ds

    xr.Dataset.from_dataframe(results_tra).to_netcdf(
        os.path.join(day_output_dir, "trajectories.nc")
    )
    results_seg_ds.to_netcdf(os.path.join(day_output_dir, "segmentation.nc"))


def _run_tobac_single_day_single_phenomenon(
    day_input_dir: str,
    day_output_dir: str,
    region: Region,
    phenomenon: WeatherPhenomenon,
    phenomenon_params: Optional[WeatherPhenomenonTobacParams] = None,
    save_images: bool = False,
) -> tuple[pd.DataFrame, xr.Dataset]:
    """
    Runs the TOBAC tracking and visualization pipeline for a single day and phenomenon.
    """
    trajectories_list = []
    segmentations_list = []

    logger.debug(f"Processing {phenomenon.value} for {day_input_dir}")

    for suffix in FOLDERS_HEIGHT_SUFF:
        features_nc = os.path.join(day_input_dir, "features.nc")

        if not os.path.exists(features_nc):
            continue
        with xr.open_dataset(features_nc) as ds:
            folder_key = f"{phenomenon.value}{suffix}"
            if folder_key not in ds:
                logger.warning(f"Folder {folder_key} not found in {features_nc}")
                continue

            da = ds[folder_key].load()
        datetimes = [pd.Timestamp(t) for t in da.time.values]

        referenced_data = build_referenced_data_from_xarray(
            da, datetimes, region_bounds=region.value
        )
        dxy, dt = get_grid_spacings(referenced_data)
        referenced_data_norm = normalize_referenced_data(referenced_data)

        if phenomenon_params is None:
            phenomenon_params = WeatherPhenomenonTobacParams[phenomenon.name]

        detection_params = phenomenon_params.value

        min_blob_size = detection_params.get("min_blob_size", 100)
        target = detection_params.get("target", "maximum")
        smooth = detection_params.get("smooth", DEFAULT_SMOOTH)
        threshold = detection_params.get("threshold", 0.6)

        # Feature detection & tracking
        features, features_weighted_points = detect_features(
            referenced_data_norm,
            threshold=threshold,
            target=target,
            smooth=smooth,
            min_blob_size=min_blob_size,
            min_distance=DEFAULT_MIN_DISTANCE,
            dxy=dxy,
        )
        v_max_at_height = DEFAULT_V_MAX_AT_HEIGHT.get(suffix, 60)
        trajectories = track_features(
            features_weighted_points,
            referenced_data,
            dt=dt,
            dxy=dxy,
            v_max=v_max_at_height,
            memory=DEFAULT_GAP_FRAMES,
        )

        # Segmentation
        segments_all = segment_features(
            features,
            referenced_data_norm,
            threshold=threshold,
            target=target,
            smooth=smooth,
            dxy=dxy,
        )

        if x := [s[1] for s in segments_all if s[1] is not None]:
            seg_da = xr.concat(x, dim="time")
            seg_da = seg_da.rename(f"{phenomenon.value}{suffix}")
            segmentations_list.append(seg_da)
            del x

        if save_images:
            from features_detection.plotting import generate_all_plots

            height_output_dir = os.path.join(day_output_dir, folder_key)
            generate_all_plots(
                da=da,
                output_dir=height_output_dir,
                cmap=detection_params.get("cmap", "viridis"),
                region=region,
                segments_all=segments_all,
                trajectories=trajectories,
            )

        if trajectories is not None and not trajectories.empty:
            tmp = trajectories.drop(
                columns=[
                    "frame",
                    "idx",
                    "threshold_value",
                    "feature",
                    "timestr",
                    "y",
                    "x",
                ],
                errors="ignore",
            )
            tmp["height"] = f"{phenomenon.value}{suffix}"
            tmp["height"] = tmp["height"].astype("category")
            trajectories_list.append(tmp)

        del trajectories
        del segments_all

    if segmentations_list:
        segmentation_ds = xr.merge(segmentations_list, compat="override", join="outer")
        del segmentations_list
    else:
        segmentation_ds = xr.Dataset()

    if trajectories_list:
        valid_dfs = [df for df in trajectories_list if not df.empty]
        if valid_dfs:
            trajectories_df = pd.concat(valid_dfs, ignore_index=True)
        else:
            trajectories_df = None
        del trajectories_list
    else:
        trajectories_df = None

    if trajectories_df is None:
        trajectories_df = pd.DataFrame(
            columns=[
                "hdim_1",
                "hdim_2",
                "num",
                "time",
                "latitude",
                "longitude",
                "cell",
                "time_cell",
                "height",
            ]
        )

    plt.close("all")
    return trajectories_df, segmentation_ds


def _extract_winds(day_input_dir: str) -> xr.Dataset:
    features_nc = os.path.join(day_input_dir, "features.nc")
    tmp_ds = xr.Dataset()
    with xr.open_dataset(features_nc) as feat_ds:
        wind_vars = [v for v in feat_ds.data_vars if "wind" in v]
        if wind_vars:
            wind_ds = feat_ds[wind_vars].load()
            tmp_ds = xr.merge([tmp_ds, wind_ds], compat="override", join="outer")

    return tmp_ds
