from __future__ import annotations


class Region:
    FVG: Region
    ITALY: Region

    def __init__(
        self,
        name: str,
        value: list[float],
        cities: dict | None = None,
        city_radius: float = 3.0,
    ):
        self.name = name
        self.value = value
        self.cities = cities
        self.city_radius = city_radius

    def __repr__(self) -> str:
        return f"Region({self.name}, {self.value}, radius={self.city_radius})"

    def get_cities(self) -> list[tuple[str, float, float]]:
        cities_dict = self.cities if self.cities is not None else CITIES
        lon_min, lon_max, lat_min, lat_max = self.value
        return [
            (city, info["lat"], info["lon"])
            for city, info in cities_dict.items()
            if lat_min <= info["lat"] <= lat_max and lon_min <= info["lon"] <= lon_max
        ]

    @classmethod
    def custom(
        cls, bounds: list[float], cities: dict | None = None, city_radius: float = 3.0
    ) -> Region:
        if len(bounds) != 4:
            raise ValueError(
                f"Region bounds must be 4 floats [lon_min, lon_max, lat_min, lat_max], got {bounds}"
            )
        return cls("CUSTOM", [float(b) for b in bounds], cities, city_radius)

    @classmethod
    def from_config(
        cls, value, cities: dict | None = None, city_radius: float = 3.0
    ) -> Region:
        if isinstance(value, dict):
            cities = value.get("cities", cities)
            city_radius = float(value.get("range", city_radius))
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
            if cities is not None or city_radius != 3.0:
                return cls(preset.name, preset.value, cities, city_radius)
            return preset
        if isinstance(value, list) and len(value) == 4:
            return cls.custom(value, cities, city_radius)
        raise ValueError(
            f"Invalid region: {value}. Must be a string, a list of 4 floats, or a dict."
        )


Region.FVG = Region("FVG", [11, 15, 44.5, 48])
Region.ITALY = Region("ITALY", [10, 16, 42, 48])

_PRESETS = {
    "FVG": Region.FVG,
    "ITALY": Region.ITALY,
}

CITIES = {
    "Barcis": {"lat": 46.1906756, "lon": 12.5544384},
    "Sappada_forni_Villa": {"lat": 46.466194, "lon": 12.876158},
    "Pontebba_Tarvisio": {"lat": 46.503154, "lon": 13.475782},
    "Gemona_Stolivizza": {"lat": 46.302533, "lon": 13.262085},
    "Udine_Palmanova": {"lat": 46.0627018, "lon": 13.2181238},
    "Trieste": {"lat": 45.6514457, "lon": 13.7608539},
    "Gorizia": {"lat": 45.9469495, "lon": 13.5973763},
    "Lignano_Grado": {"lat": 45.80574, "lon": 13.16546},
    "Pordenone": {"lat": 45.9560557, "lon": 12.6453929},
}
