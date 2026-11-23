import itertools
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from region import Region


def overlay_cities(axs: plt.Axes, region: Region):
    """
    Overlay city markers and labels on top of the map using Cartopy.
    """

    cities = region.get_cities()
    for name, lat, lon in cities:
        axs.plot(
            [lon],
            [lat],
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
    frames: list[int],
):
    """Plots cloud centre markers and the fading trajectory path using the cell's frames up to itime."""
    for t0, t1 in itertools.pairwise(frames):
        line = track[(track["frame"] == t0) | (track["frame"] == t1)]
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
    f_weighted: pd.DataFrame, cell_id: int, region: Region, axs: plt.Axes
):
    """Renders cell ID text annotation near the point natively."""
    if f_weighted.empty or "longitude" not in f_weighted:
        return

    lon_pos = f_weighted["longitude"].values[0]
    lat_pos = f_weighted["latitude"].values[0]

    lon_min, lon_max, _, _ = region.value
    lon_span = lon_max - lon_min

    # Adjust position so it doesn't clip off-screen
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
    output_dir: Path,
    cmap: str,
    region: Region,
    segments_all: list | tuple,
    trajectories: pd.DataFrame,
):
    """
    Renders all frames in the given DataArray with native Cartopy projection,
    overlaying segments and tracking lines, and saving to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = {} if trajectories is None else dict(tuple(trajectories.groupby("cell")))

    for i in range(da.sizes["time"]):
        frame_da = da.isel(time=i)

        valid_time_pd = pd.Timestamp(da.time.values[i])
        out_name = f"{da.name}_{valid_time_pd.strftime('%Y%m%d_%H%M')}_tracked.png"
        out_path = output_dir / out_name

        fig, axs = plt.subplots(
            figsize=(10, 8), dpi=100, subplot_kw={"projection": ccrs.PlateCarree()}
        )
        axs.set_extent(region.value, crs=ccrs.PlateCarree())

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

        axs.coastlines(resolution="10m", linewidth=1)
        axs.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")

        overlay_cities(axs, region)

        for cell_id, track in tracks.items():
            f_weighted = track[track["frame"] == i]
            if f_weighted.empty:
                continue
            frames = sorted(f for f in track["frame"].unique() if f <= i)
            symbol, color = ("^", "white") if len(frames) == 1 else ("x", "red")
            print_clouds_center_line(symbol, color, f_weighted, i, track, axs, frames)
            print_cloud_labels(f_weighted, cell_id, region, axs)

        entry = next((s for s in segments_all if s[0] == i), None)
        if entry is not None:
            _, seg_labels, _ = entry
            if seg_labels is not None:
                seg_labels2d = seg_labels.isel(time=0)
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
        plt.close(fig)
