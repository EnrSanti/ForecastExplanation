from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("ForecastExplanation")

# CERRA grid: Lambert conformal (tangent cone) on a sphere, 1069 x 1069 points
# 5.5 km apart, the whole area downloaded from the CDS
_CERRA_EARTH_RADIUS = 6371229.0
_CERRA_LAT0, _CERRA_LON0 = 50.0, 8.0  # standard parallel, central meridian
_CERRA_SW_CORNER = (20.292281, -17.485)  # (lat, lon) of the first grid point
_CERRA_SIDE = 1068 * 5500.0  # metres


def _cerra_xy(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Projected metres of the CERRA grid, up to a constant offset."""
    phi0 = np.radians(_CERRA_LAT0)
    n = np.sin(phi0)
    scale = _CERRA_EARTH_RADIUS * np.cos(phi0) / n
    ratio = np.tan(np.pi / 4 + phi0 / 2) / np.tan(np.pi / 4 + np.radians(lat) / 2)
    rho = scale * ratio**n
    theta = n * np.radians(np.asarray(lon) - _CERRA_LON0)
    return rho * np.sin(theta), -rho * np.cos(theta)


def _outside_cerra(lat, lon) -> np.ndarray:
    x, y = _cerra_xy(lat, lon)
    x0, y0 = _cerra_xy(*_CERRA_SW_CORNER)
    x, y = x - x0, y - y0
    return (x < 0) | (x > _CERRA_SIDE) | (y < 0) | (y > _CERRA_SIDE)


class Region:
    FVG: Region
    ITALY: Region

    def __init__(
        self,
        name: str,
        value: list[float],
        cities: dict | None = None,
        city_radius: float = 20.0,
    ):
        lon_min, lon_max, lat_min, lat_max = value
        if lon_min >= lon_max or lat_min >= lat_max:
            raise ValueError(
                f"region: bounds must be [lon_min, lon_max, lat_min, lat_max], "
                f"got {value}"
            )
        if city_radius <= 0:
            raise ValueError(f"region: radius must be > 0, got {city_radius}")
        self.name = name
        self.value = value
        self.cities = cities
        self.city_radius = city_radius
        self._warn_outside()

    def __repr__(self) -> str:
        return f"Region({self.name}, {self.value}, radius={self.city_radius})"

    def _contains(self, lat: float, lon: float) -> bool:
        lon_min, lon_max, lat_min, lat_max = self.value
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

    def _warn_outside(self) -> None:
        """Warns about the cities get_cities drops and about the part of the
        region with no CERRA data."""
        cities_dict = self.cities if self.cities is not None else CITIES
        outside = [
            city
            for city, info in cities_dict.items()
            if not self._contains(info["lat"], info["lon"])
        ]
        if outside:
            logger.warning(
                f"Cities outside region {self.name} {self.value}, ignored: {outside}"
            )

        lon_min, lon_max, lat_min, lat_max = self.value
        lats, lons = np.meshgrid(
            np.linspace(lat_min, lat_max, 50), np.linspace(lon_min, lon_max, 50)
        )
        if _outside_cerra(lats, lons).any():
            logger.warning(
                f"Region {self.name} {self.value} is partially outside the CERRA "
                "grid: only the part inside it has data"
            )

    def get_cities(self) -> list[tuple[str, float, float]]:
        cities_dict = self.cities if self.cities is not None else CITIES
        return [
            (city, info["lat"], info["lon"])
            for city, info in cities_dict.items()
            if self._contains(info["lat"], info["lon"])
        ]

    @classmethod
    def custom(
        cls, bounds: list[float], cities: dict | None = None, city_radius: float = 20.0
    ) -> Region:
        if len(bounds) != 4:
            raise ValueError(
                f"Region bounds must be 4 floats [lon_min, lon_max, lat_min, lat_max], got {bounds}"
            )
        return cls("CUSTOM", [float(b) for b in bounds], cities, city_radius)

    @classmethod
    def from_config(
        cls, value, cities: dict | None = None, city_radius: float = 20.0
    ) -> Region:
        if isinstance(value, dict):
            cities = value.get("cities", cities)
            city_radius = float(value.get("radius", city_radius))
            if "name" in value:
                return cls.from_config(value["name"], cities, city_radius)
            elif "bounds" in value:
                return cls.from_config(value["bounds"], cities, city_radius)
            else:
                raise ValueError(
                    f"Invalid region dict: {value}. Must contain 'name' or 'bounds'."
                )

        if isinstance(value, str):
            name = value.upper()
            preset = _PRESETS.get(name)
            if preset is None:
                raise ValueError(
                    f"Unknown region '{name}'. Must be one of {list(_PRESETS.keys())} or a list of 4 floats."
                )
            if cities is not None or city_radius != 20.0:
                return cls(preset.name, preset.value, cities, city_radius)
            return preset
        if isinstance(value, list) and len(value) == 4:
            return cls.custom(value, cities, city_radius)
        raise ValueError(
            f"Invalid region: {value}. Must be a string, a list of 4 floats, or a dict."
        )


CITIES = {
    "Prealpi Carniche": {"lat": 46.1906756, "lon": 12.5544384},
    "Alpi Carniche": {"lat": 46.466194, "lon": 12.876158},
    "Alpi Giulie": {"lat": 46.503154, "lon": 13.475782},
    "Prealpi Giulie": {"lat": 46.302533, "lon": 13.262085},
    "Udine": {"lat": 46.0627018, "lon": 13.2181238},
    "Trieste": {"lat": 45.6514457, "lon": 13.7608539},
    "Gorizia": {"lat": 45.9469495, "lon": 13.5973763},
    "Lignano": {"lat": 45.80574, "lon": 13.16546},
    "Pordenone": {"lat": 45.9560557, "lon": 12.6453929},
}

Region.FVG = Region("FVG", [11, 15, 44.5, 48])
Region.ITALY = Region("ITALY", [10, 16, 42, 48])

_PRESETS = {"FVG": Region.FVG, "ITALY": Region.ITALY}
