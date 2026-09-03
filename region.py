from __future__ import annotations


class Region:
    FVG: Region
    ITALY: Region

    def __init__(self, name: str, value: list[float]):
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"Region({self.name}, {self.value})"

    @classmethod
    def custom(cls, bounds: list[float]) -> Region:
        if len(bounds) != 4:
            raise ValueError(f"Region bounds must be 4 floats [lon_min, lon_max, lat_min, lat_max], got {bounds}")
        return cls("CUSTOM", [float(b) for b in bounds])

    @classmethod
    def from_config(cls, value) -> Region:
        if isinstance(value, str):
            name = value.upper()
            preset = _PRESETS.get(name)
            if preset is None:
                raise ValueError(f"Unknown region '{name}'. Must be one of {list(_PRESETS.keys())} or a list of 4 floats.")
            return preset
        if isinstance(value, list) and len(value) == 4:
            return cls.custom(value)
        raise ValueError(f"Invalid region: {value}. Must be a string or a list of 4 floats.")


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