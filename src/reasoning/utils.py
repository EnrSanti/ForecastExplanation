import xarray as xr


def get_heights(data: xr.Dataset) -> list[str]:
    """Level suffixes ("0300m", ...) present in the data, ascending."""
    prefix = "wind_direction_at_"
    return sorted(
        str(v).removeprefix(prefix) for v in data.data_vars if str(v).startswith(prefix)
    )


def pixel_area_km2(data: xr.Dataset) -> float:
    """Area of one grid cell, from the dataset's `dxy` (grid spacing in m)."""
    return (float(data.attrs["dxy"]) / 1000.0) ** 2
