import logging
import re
from typing import Callable, NamedTuple, Optional

import pymupdf
import requests

ZONES = ["a1", "a2", "a3", "a4", "z2", "z4"]
NAMES = {
    "regione": "regione",
    "z1": "monti",
    "a1": "Alpi carniche",
    "a2": "Alpi giulie",
    "a3": "Prealpi carniche",
    "a4": "Prealpi giulie",
    "z2": "alta pianura",
    "z3": "bassa pianura",
    "z4": "costa",
}
BASE_ARCHIVE_URL = "https://www.osmer.fvg.it/ajax/getPrevisArchive.php"
PDF_BASE = "https://www.osmer.fvg.it/"
logger = logging.getLogger("ForecastExplanation")


def text_extract(dt) -> dict:
    session = requests.Session()
    yyyymmdd = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"
    data = {}

    for zone in ZONES:
        params = {
            "a": dt.year,
            "m": dt.month,
            "g": dt.day,
            "z": zone,
            "l": "it",
            "ln": "",
        }

        r = session.get(BASE_ARCHIVE_URL, params=params, timeout=10)
        if r.status_code != 200:
            logger.error(f"Failed to load archive HTML for {zone}, date {dt}")
            continue

        clean_html = r.text.replace("\\/", "/").replace('\\"', '"')
        pattern = rf"pdf/\d{{4}}/\d{{8}}/(\d{{12}})/pdf/{zone}-\d{{8}}-it\.pdf"
        matches = re.findall(pattern, clean_html)

        if not matches:
            logger.error(f"No PDF link found for {zone}, date {dt}")
            continue

        # most recent timestamp
        timestamp = matches[-1]
        pdf_url = f"{PDF_BASE}pdf/{dt.year}/{yyyymmdd}/{timestamp}/pdf/{zone}-{yyyymmdd}-it.pdf"

        pdf_resp = session.get(pdf_url, timeout=10)
        if pdf_resp.status_code != 200:
            logger.error(f"Failed to download PDF from {pdf_url}")
            continue

        doc = pymupdf.open(stream=pdf_resp.content, filetype="pdf")
        for page in doc:
            text = page.get_text("text").strip()
            data[NAMES[zone]] = extract_zone_data(text, zone)
            break

    return data


class FieldSpec(NamedTuple):
    label: str
    keys: tuple[str, ...]
    casts: Optional[dict[str, Callable]] = None


def apply_fields(lines: list[str], specs: list[FieldSpec]) -> dict:
    """Extract fields from a list of specs, each matched against a label line."""
    data = {}
    for i, line in enumerate(lines):
        for label, keys, casts in specs:
            if label not in line or keys[0] in data:
                continue

            for offset, key in enumerate(keys, start=1):
                if i + offset >= len(lines):
                    break
                cast = (casts or {}).get(key, str)
                data[key] = cast(lines[i + offset])

    return data


MOUNTAIN_COMMON_FIELDS = [
    FieldSpec("Temperatura media a 1.000 m (°C)", ("temperatura_media_1000",)),
    FieldSpec("Temperatura media a 2.000 m (°C)", ("temperatura_media_2000",)),
    FieldSpec("Probabilità precipitazioni estese (%)", ("pioggia_prob",)),
    FieldSpec("Probabilità di temporali (%)", ("temporale_prob",)),
    FieldSpec("Quota zero termico (m)", ("quota_zero_termico",)),
    FieldSpec("Quota delle nevicate (m)", ("quota_nevicate",)),
    FieldSpec(
        "Vento medio a 2.000 m (m/s)",
        ("vento_2000_direzione", "vento_2000_velocita"),
        {"vento_2000_velocita": float},
    ),
    FieldSpec(
        "Vento medio a 3.000 m (m/s)",
        ("vento_3000_direzione", "vento_3000_velocita"),
        {"vento_3000_velocita": float},
    ),
]

ZONE_FIELDS = {
    "a1": MOUNTAIN_COMMON_FIELDS
    + [
        FieldSpec("Forni Avoltri", ("forni_avoltri_min", "forni_avoltri_max")),
        FieldSpec("M. Zoncolan", ("m_zoncolan_min", "m_zoncolan_max")),
    ],
    "a2": MOUNTAIN_COMMON_FIELDS
    + [
        FieldSpec("Tarvisio", ("tarvisio_min", "tarvisio_max")),
        FieldSpec("M. Lussari", ("m_lussari_min", "m_lussari_max")),
    ],
    "a3": MOUNTAIN_COMMON_FIELDS
    + [
        FieldSpec("Claut", ("claut_min", "claut_max")),
        FieldSpec("Piancavallo", ("piancavallo_min", "piancavallo_max")),
    ],
    "a4": MOUNTAIN_COMMON_FIELDS
    + [
        FieldSpec("Sella Nevea", ("sella_nevea_min", "sella_nevea_max")),
        FieldSpec("M. Canin (R. Gilberti)", ("m_canin_min", "m_canin_max")),
    ],
    "z2": [
        FieldSpec("Temperatura minima pianura (°C)", ("temperatura_minima_pianura",)),
        FieldSpec("Temperatura massima pianura (°C)", ("temperatura_massima_pianura",)),
        FieldSpec("Quota zero termico (m)", ("quota_zero_termico",)),
        FieldSpec("Quota delle nevicate (m)", ("quota_nevicate",)),
        FieldSpec(
            "Probabilità precipitazioni estese (%)",
            ("pioggia_prob_prealpi", "pioggia_prob_pianura"),
        ),
        FieldSpec(
            "Probabilità di temporali (%)",
            ("temporale_prob_prealpi", "temporale_prob_pianura"),
        ),
    ],
    "z4": [
        FieldSpec("Temperatura minima (°C)", ("temperatura_minima",)),
        FieldSpec("Temperatura massima (°C)", ("temperatura_massima",)),
        FieldSpec("Probabilità precipitazioni estese (%)", ("pioggia_prob",)),
        FieldSpec("Probabilità di temporali (%)", ("temporale_prob",)),
    ],
}


def extract_zone_data(raw_text, zone):
    main_forecast = raw_text.split("ARPA FVG")[0]
    lines = [line.strip() for line in main_forecast.split("\n") if line.strip()]

    fields = ZONE_FIELDS.get(zone)
    if fields is None:
        logger.warning(f"Unknown zone {zone}")
        return {}

    data = apply_fields(lines, fields)

    if zone == "z4":
        data.update(extract_costa_wind(lines))

    return data


def extract_costa_wind(lines):
    data = {}

    for i, line in enumerate(lines):
        if "Vento medio al largo: direzione ed intensità (kt)" not in line:
            continue
        if "vento_mattino_direzione" in data:
            break

        if i + 2 < len(lines):
            mattino = lines[i + 2].split()
            if len(mattino) >= 2:
                data["vento_mattino_direzione"] = mattino[0]
                data["vento_mattino_intensita"] = mattino[1]

        if i + 4 < len(lines):
            pomeriggio = lines[i + 4].split()
            if len(pomeriggio) >= 2:
                data["vento_pomeriggio_direzione"] = pomeriggio[0]
                data["vento_pomeriggio_intensita"] = pomeriggio[1]

    return data
