import logging
import os
from typing import Set, Dict, List

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from region import Region, CITIES

logger = logging.getLogger(__name__)


def overlay_cities(axs: plt.Axes, region: Region = None):
    """
    Overlay city markers and labels on top of the map using Cartopy.
    """

    if region is not None:
        cities = region.get_cities()
    else:
        cities = CITIES

    for name, lon, lat in cities:
        axs.plot(
            lon,
            lat,
            marker="o",
            markersize=3,
            color="red",
            markeredgecolor="white",
            markeredgewidth=0.5,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        axs.annotate(
            name,
            xy=(lon, lat),
            xytext=(4, -4),
            textcoords="offset points",
            color="red",
            fontsize=6,
            zorder=11,
            transform=ccrs.PlateCarree(),
        )


def print_clouds_center_line(
    printing_symbol: str,
    color: str,
    f_weighted: pd.DataFrame,
    itime: int,
    track: pd.DataFrame,
    axs: plt.Axes,
    cell_id: int,
    persisted_cells: Set[int],
    all_frames_for_cell: Dict[int, List[int]],
):
    """Plots cloud center markers and fading trajectory path natively."""

    if cell_id in persisted_cells:
        frames_list = all_frames_for_cell.get(int(cell_id), [])
        if len(frames_list) >= 2:
            last_frame = frames_list[-2]
            line = track[(track["frame"] == last_frame) | (track["frame"] == itime)]
            if not line.empty:
                axs.plot(
                    line["longitude"],
                    line["latitude"],
                    color="cyan",
                    linewidth=2.5,
                    alpha=0.7,
                    transform=ccrs.PlateCarree(),
                    path_effects=[
                        pe.Stroke(linewidth=4, foreground="black"),
                        pe.Normal(),
                    ],
                )

    try:
        frames = all_frames_for_cell.get(int(cell_id), [])
        for t0, t1 in zip(frames[:-1], frames[1:]):
            line = track[(track["frame"] == t0) | (track["frame"] == t1)]
            if not line.empty:
                time_diff = itime - track.iloc[0].frame
                alpha = 0.3 + 0.7 * (t0 - track.iloc[0].frame) / (
                    time_diff if time_diff != 0 else 1
                )
                alpha = max(0.2, min(1.0, alpha))
                axs.plot(
                    line["longitude"],
                    line["latitude"],
                    color="cyan",
                    linewidth=2.5,
                    alpha=alpha,
                    transform=ccrs.PlateCarree(),
                    path_effects=[
                        pe.Stroke(linewidth=4, foreground="black", alpha=alpha),
                        pe.Normal(),
                    ],
                )
    except (KeyError, IndexError) as e:
        logger.debug(f"Failed to draw trail for cell {cell_id}: {e}")

    if not f_weighted.empty:
        axs.scatter(
            f_weighted["longitude"],
            f_weighted["latitude"],
            s=40,
            color=color,
            marker=printing_symbol,
            zorder=5,
            transform=ccrs.PlateCarree(),
        )


def print_cloud_labels(
    f_weighted: pd.DataFrame,
    cell_id: int,
    region: Region,
    axs: plt.Axes,
):
    """Renders cell ID text annotation near the point natively."""
    if f_weighted.empty or "longitude" not in f_weighted:
        return

    lon_pos = f_weighted["longitude"].values[0]
    lat_pos = f_weighted["latitude"].values[0]

    lon_min, lon_max, _, _ = region.value
    lon_span = lon_max - lon_min

    # Adjust position so it doesn't clip off screen
    if lon_pos < lon_min + 0.05 * lon_span:
        lon_pos += 0.02 * lon_span
    if lon_pos > lon_max - 0.05 * lon_span:
        lon_pos -= 0.02 * lon_span

    axs.annotate(
        str(int(cell_id)),
        xy=(lon_pos, lat_pos),
        xytext=(0, 10),
        textcoords="offset points",
        color="red",
        fontsize=6,
        fontweight="bold",
        zorder=12,
        transform=ccrs.PlateCarree(),
    )


def generate_all_plots(
    da: xr.DataArray,
    output_dir: str,
    cmap: str,
    region: Region,
    segments_all: list,
    trajectories: pd.DataFrame,
):
    """
    Renders all frames in the given DataArray with native Cartopy projection,
    overlaying segments and tracking lines, and saving to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build tracking info memory mapping
    cell_info_by_frame = {}
    if trajectories is not None and not trajectories.empty:
        frames_by_cell = trajectories.groupby("cell")["frame"].apply(
            lambda s: sorted(s.unique())
        )

        for i, frame_traj in trajectories.groupby("frame"):
            cell_ids = set(frame_traj["cell"].dropna().unique())
            persisted, new_cells, all_frames_for_cell = set(), set(), {}
            for cell_id in cell_ids:
                cell_frames = frames_by_cell.get(cell_id, [])
                idx = cell_frames.index(i) if i in cell_frames else -1
                if idx > 0:
                    persisted.add(cell_id)
                else:
                    new_cells.add(cell_id)
                all_frames_for_cell[int(cell_id)] = cell_frames[: idx + 1]

            cell_info_by_frame[i] = {
                "cell_ids": cell_ids,
                "persisted": persisted,
                "new_cells": new_cells,
                "all_frames_for_cell": all_frames_for_cell,
            }

    trajectories_by_cell = {}
    if trajectories is not None and not trajectories.empty:
        for cell_id, df in trajectories.groupby("cell"):
            trajectories_by_cell[cell_id] = df

    for i in range(da.sizes["time"]):
        frame_da = da.isel(time=i)

        # Valid time to filename string matching old pipeline format: e.g., ..._20090102_0100_tracked.png
        valid_time_pd = pd.Timestamp(da.time.values[i])
        out_name = f"{da.name}_{valid_time_pd.strftime('%Y%m%d_%H%M')}_tracked.png"
        out_path = os.path.join(output_dir, out_name)

        # 10x8 inches matches Cartopy default map size used in old _save_scalar_maps
        fig, axs = plt.subplots(
            figsize=(10, 8), dpi=100, subplot_kw={"projection": ccrs.PlateCarree()}
        )
        axs.set_extent(region.value, crs=ccrs.PlateCarree())

        # Map the underlying values
        vmin, vmax = 0, 1  # features.nc is already normalized 0..1
        axs.pcolormesh(
            frame_da["longitude"],
            frame_da["latitude"],
            frame_da,
            cmap=cmap,
            shading="gouraud",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )

        # Add coastlines and borders natively
        axs.coastlines(resolution="10m", linewidth=1)
        axs.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")

        # Add cities
        overlay_cities(axs)

        # Add tracking data
        info = cell_info_by_frame.get(
            i,
            {
                "cell_ids": set(),
                "persisted": set(),
                "new_cells": set(),
                "all_frames_for_cell": {},
            },
        )
        cell_ids = info["cell_ids"]
        persisted = info["persisted"]
        new_cells = info["new_cells"]
        all_frames_for_cell = info["all_frames_for_cell"]

        for cell_id in cell_ids:
            track = trajectories_by_cell.get(cell_id, pd.DataFrame())
            f_weighted = track[track["frame"] == i]

            printing_symbol, color = (
                ("^", "white") if cell_id in new_cells else ("x", "red")
            )
            print_clouds_center_line(
                printing_symbol,
                color,
                f_weighted,
                i,
                track,
                axs,
                cell_id,
                persisted,
                all_frames_for_cell,
            )

            if len(f_weighted["longitude"]) > 0:
                print_cloud_labels(f_weighted, cell_id, region, axs)

        # Draw Segments
        entry = next((s for s in segments_all if s[0] == i), None)
        if entry is not None:
            _, seg_labels, _ = entry
            if seg_labels is not None:
                seg_labels2d = seg_labels.isel(time=0)
                # Plot contours mapping indices back to lat/lon by providing coords
                axs.contour(
                    seg_labels2d["longitude"],
                    seg_labels2d["latitude"],
                    seg_labels2d.values,
                    levels=[0.5],
                    colors="k",
                    transform=ccrs.PlateCarree(),
                )

        axs.set_title("")
        axs.axis("off")

        fig.canvas.draw()
        bbox = axs.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out_path, dpi=100, bbox_inches=bbox, pad_inches=0, transparent=True)
        fig.clf()
        plt.close(fig)
