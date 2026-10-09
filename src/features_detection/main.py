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

from features_detection.constants import (
    DEFAULT_GAP_FRAMES,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_SMOOTH,
    DEFAULT_V_MAX_AT_HEIGHT,
    FOLDERS_HEIGHT_SUFF,
    RAW_FEATURES_VARS,
    TOBAC_PHENOMENA,
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
    to_compressed_netcdf,
)
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
        results = [
            _run_tobac_single_day_single_phenomenon(
                feat_ds, day_output_dir, region, phenomenon, save_images
            )
            for phenomenon in TOBAC_PHENOMENA
        ]
        _create_output_features_nc(feat_ds, day_output_dir)

    dfs = [df for df, _ in results if not df.empty]
    results_tra = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    dss = [ds for _, ds in results if ds.data_vars]
    results_seg_ds = xr.merge(dss, compat="override", join="outer")

    xr.Dataset.from_dataframe(results_tra).to_netcdf(day_output_dir / "trajectories.nc")
    to_compressed_netcdf(results_seg_ds, day_output_dir / "segmentation.nc")


def _run_tobac_single_day_single_phenomenon(
    feat_ds: xr.Dataset,
    day_output_dir: Path,
    region: Region,
    phenomenon: WeatherPhenomenon,
    save_images: bool = False,
) -> tuple[pd.DataFrame, xr.Dataset]:
    """
    Runs the TOBAC tracking and visualization pipeline for a single day and phenomenon.
    """
    trajectories_list = []
    segmentations_list = []
    detection_params = WeatherPhenomenonTobacParams[phenomenon.name].value

    logger.debug(f"Processing {phenomenon.value} for {day_output_dir}")

    for suffix in FOLDERS_HEIGHT_SUFF:
        folder_key = f"{phenomenon.value}{suffix}"
        if folder_key not in feat_ds:
            logger.warning(
                f"Folder {folder_key} not found in {feat_ds.encoding.get('source')}"
            )
            continue

        da = feat_ds[folder_key].load()
        datetimes = [pd.Timestamp(t) for t in da.time.values]

        referenced_data = build_referenced_data_from_xarray(da, datetimes)
        dxy, dt = get_grid_spacings(referenced_data)

        min_blob_size = int(detection_params.get("min_blob_size", 100))
        target = str(detection_params.get("target", "maximum"))
        smooth = float(detection_params.get("smooth", DEFAULT_SMOOTH))
        threshold = float(detection_params.get("threshold", 0.6))

        # Feature detection & tracking
        features, features_weighted_points = detect_features(
            referenced_data,
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
            referenced_data,
            threshold=threshold,
            target=target,
            smooth=smooth,
            dxy=dxy,
        )

        if x := [s[1] for s in segments_all if s[1] is not None]:
            seg_da = xr.concat(x, dim="time")
            segmentations_list.append(seg_da.rename(folder_key))

        if save_images:
            from features_detection.plotting import generate_all_plots

            height_output_dir = day_output_dir / folder_key
            generate_all_plots(
                da=da,
                output_dir=height_output_dir,
                cmap=str(detection_params.get("cmap", "viridis")),
                region=region,
                segments_all=segments_all or [],
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

    if segmentations_list:
        segmentation_ds = xr.merge(segmentations_list, compat="override", join="outer")
    else:
        segmentation_ds = xr.Dataset()

    if trajectories_list:
        trajectories_df = pd.concat(trajectories_list, ignore_index=True)
    else:
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


def _create_output_features_nc(feat_ds: xr.Dataset, day_output_dir: Path) -> None:
    """Raw fields for reasoning (wind, raw_*, cloud cover in %) with dxy."""
    raw_vars = [
        v
        for v in feat_ds.data_vars
        if any(prefix in str(v) for prefix in RAW_FEATURES_VARS)
    ]
    out = feat_ds[raw_vars].load()
    for v in feat_ds.data_vars:
        if str(v).startswith("cloud_at_"):
            out[f"raw_{v}"] = feat_ds[v].load() * 100

    da = feat_ds[raw_vars[0]]
    ref_data = build_referenced_data_from_xarray(da, list(da.time.values))
    out.attrs["dxy"] = get_grid_spacings(ref_data)[0]
    to_compressed_netcdf(out, day_output_dir / "features.nc")
