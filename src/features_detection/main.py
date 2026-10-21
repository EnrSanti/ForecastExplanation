import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import matplotlib
import pandas as pd
import xarray as xr
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_extraction import FOLDERS, to_compressed_netcdf
from features_detection.constants import (
    DT,
    GAP_FRAMES,
    MIN_DISTANCE,
    RAW_FEATURES_VARS,
    TOBAC_PARAMS,
    V_MAX_AT_LEVEL,
)
from features_detection.features import (
    detect_features,
    segment_features,
    track_features,
)
from features_detection.utils import build_referenced_data_from_xarray, grid_spacing
from region import Region

logger = logging.getLogger(__name__)


def run_tobac(
    dates: list[date],
    input_dir: Path,
    output_dir: Path,
    region: Region,
    force: bool = False,
    save_images: bool = False,
    workers: int = 12,
) -> list[date]:
    """
    Executes TOBAC tracking across the specified list of dates and weather phenomena.
    """

    logger.info("Starting TOBAC.")
    output_dir.mkdir(parents=True, exist_ok=True)
    ok = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_tobac_single_day,
                target_date,
                input_dir,
                output_dir,
                region,
                force=force,
                save_images=save_images,
            ): target_date
            for target_date in dates
        }

        for future in tqdm(
            as_completed(futures), total=len(dates), desc="TOBAC Processing"
        ):
            target_date = futures[future]
            try:
                future.result()
                ok.append(target_date)
            except Exception:
                logger.exception(f"TOBAC failed for {target_date}")

    logger.info("TOBAC runs completed.")
    return sorted(ok)


def _run_tobac_single_day(
    target_date: date,
    input_dir: Path,
    output_dir: Path,
    region: Region,
    force: bool = False,
    save_images: bool = False,
) -> None:
    day_input_dir = input_dir / target_date.strftime("%Y-%m-%d")
    day_output_dir = output_dir / target_date.strftime("%Y-%m-%d")
    day_output_dir.mkdir(parents=True, exist_ok=True)

    if not force and (day_output_dir / "segmentation.nc").exists():
        logger.debug(
            f"Segmentation already exists for {target_date.strftime('%Y-%m-%d')}. Skipping."
        )
        return

    features_nc = day_input_dir / "features.nc"
    if not features_nc.exists():
        raise FileNotFoundError(f"Missing TOBAC input {features_nc}")

    with xr.open_dataset(features_nc, engine="h5netcdf") as feat_ds:
        dxy = grid_spacing(feat_ds)
        segmentations, trajectories = [], []
        for phenomenon, params in TOBAC_PARAMS.items():
            segs, tracks = _run_tobac_single_day_single_phenomenon(
                feat_ds, day_output_dir, region, phenomenon, params, dxy, save_images
            )
            segmentations += segs
            trajectories += tracks
        _create_output_features_nc(feat_ds, dxy, day_output_dir)

    tracks_df = (
        pd.concat(trajectories, ignore_index=True) if trajectories else pd.DataFrame()
    )
    xr.Dataset.from_dataframe(tracks_df).to_netcdf(day_output_dir / "trajectories.nc")
    segmentation = xr.merge(segmentations, compat="override", join="outer")
    to_compressed_netcdf(segmentation, day_output_dir / "segmentation.nc")


def _run_tobac_single_day_single_phenomenon(
    feat_ds: xr.Dataset,
    day_output_dir: Path,
    region: Region,
    phenomenon: str,
    params: dict,
    dxy: float,
    save_images: bool = False,
) -> tuple[list[xr.DataArray], list[pd.DataFrame]]:
    """
    Runs the TOBAC tracking and visualization pipeline for a single day and
    phenomenon. Returns the segmentation mask of every height and the tracks
    of the heights that have any.
    """
    trajectories_list = []
    segmentations_list = []
    logger.debug(f"Processing {phenomenon} for {day_output_dir}")

    for level, suffix in FOLDERS.items():
        folder_key = f"{phenomenon}{suffix}"
        if folder_key not in feat_ds:
            logger.warning(
                f"Folder {folder_key} not found in {feat_ds.encoding.get('source')}"
            )
            continue

        da = feat_ds[folder_key].load()
        datetimes = [pd.Timestamp(t) for t in da.time.values]

        referenced_data = build_referenced_data_from_xarray(da, datetimes)

        threshold, target, smooth = (
            params["threshold"],
            params["target"],
            params["smooth"],
        )

        # Feature detection & tracking
        features, features_weighted_points = detect_features(
            referenced_data,
            threshold=threshold,
            target=target,
            smooth=smooth,
            min_blob_size=params["min_blob_size"],
            min_distance=MIN_DISTANCE,
            dxy=dxy,
        )
        trajectories = track_features(
            features_weighted_points,
            dt=DT,
            v_max=V_MAX_AT_LEVEL[level],
            memory=GAP_FRAMES,
        )

        # Segmentation
        segments_all = segment_features(
            features,
            referenced_data,
            threshold=threshold,
            target=target,
            smooth=smooth,
            dxy=dxy,
        )

        if x := [s[1] for s in segments_all if s[1] is not None]:
            seg_da = xr.concat(x, dim="time")
        else:  # nothing detected all day: an all-background mask
            seg_da = xr.zeros_like(referenced_data, dtype=int)
        segmentations_list.append(seg_da.rename(folder_key))

        if save_images:
            from features_detection.plotting import generate_all_plots

            height_output_dir = day_output_dir / folder_key
            generate_all_plots(
                da=da,
                output_dir=height_output_dir,
                cmap=params["cmap"],
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
            tmp["height"] = folder_key
            tmp["height"] = tmp["height"].astype("category")
            trajectories_list.append(tmp)

    plt.close("all")
    return segmentations_list, trajectories_list


def _create_output_features_nc(
    feat_ds: xr.Dataset, dxy: float, day_output_dir: Path
) -> None:
    """Raw fields for reasoning (wind, raw_*) with dxy."""
    raw_vars = [
        v
        for v in feat_ds.data_vars
        if any(prefix in str(v) for prefix in RAW_FEATURES_VARS)
    ]
    out = feat_ds[raw_vars].load()
    out.attrs["dxy"] = dxy
    to_compressed_netcdf(out, day_output_dir / "features.nc")
