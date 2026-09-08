import logging
import re
from datetime import datetime

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


def text_extraction(dt):
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

        try:
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
        except Exception as e:
            logger.error(f"Error processing {zone}: {e}")

    return data


def extract_zone_data(raw_text, zone):
    main_forecast = raw_text.split("ARPA FVG")[0]
    lines = [line.strip() for line in main_forecast.split("\n") if line.strip()]

    match zone:
        case "a1":
            data = handler_alpi_carniche(lines)
        case "a2":
            data = handler_alpi_giulie(lines)
        case "a3":
            data = handler_prealpi_carniche(lines)
        case "a4":
            data = handler_prealpi_giulie(lines)
        case "z2":
            data = handler_alta_pianura(lines)
        case "z4":
            data = handler_costa(lines)
        case _:
            print("Unknown zone")
            data = {}

    return data


def handler_alpi_carniche(lines):
    data = {}
    for i, line in enumerate(lines):
        if (
            "Temperatura media a 1.000 m (°C)" in line
            and "temperatura_media_1000" not in data
        ):
            data["temperatura_media_1000"] = lines[i + 1]

        if (
            "Temperatura media a 2.000 m (°C)" in line
            and "temperatura_media_2000" not in data
        ):
            data["temperatura_media_2000"] = lines[i + 1]

        if (
            "Probabilità precipitazioni estese (%)" in line
            and "pioggia_prob" not in data
        ):
            data["pioggia_prob"] = lines[i + 1]

        if "Probabilità di temporali (%)" in line and "temporale_prob" not in data:
            data["temporale_prob"] = lines[i + 1]

        if "Quota zero termico (m)" in line and "quota_zero_termico" not in data:
            data["quota_zero_termico"] = int(lines[i + 1])

        if "Quota delle nevicate (m)" in line and "quota_nevicate" not in data:
            data["quota_nevicate"] = int(lines[i + 1])

        if "Vento medio a 2.000 m (m/s)" in line and "vento_2000" not in data:
            data["vento_2000_direzione"] = lines[i + 1]
            data["vento_2000_velocita"] = float(lines[i + 2])

        if "Vento medio a 3.000 m (m/s)" in line and "vento_3000" not in data:
            data["vento_3000_direzione"] = lines[i + 1]
            data["vento_3000_velocita"] = float(lines[i + 2])

        if "Forni Avoltri" in line and "forni_avoltri_min" not in data:
            data["forni_avoltri_min"] = int(lines[i + 1])
            data["forni_avoltri_max"] = int(lines[i + 2])

        if "M. Zoncolan" in line and "m_zoncolan_min" not in data:
            data["m_zoncolan_min"] = int(lines[i + 1])
            data["m_zoncolan_max"] = int(lines[i + 2])

    return data


def handler_alpi_giulie(lines):
    data = {}
    for i, line in enumerate(lines):
        if (
            "Temperatura media a 1.000 m (°C)" in line
            and "temperatura_media_1000" not in data
        ):
            data["temperatura_media_1000"] = int(lines[i + 1])

        if (
            "Temperatura media a 2.000 m (°C)" in line
            and "temperatura_media_2000" not in data
        ):
            data["temperatura_media_2000"] = int(lines[i + 1])

        if (
            "Probabilità precipitazioni estese (%)" in line
            and "pioggia_prob" not in data
        ):
            value = lines[i + 1].strip()
            data["pioggia_prob"] = int(value) if value.isdigit() else value

        if "Probabilità di temporali (%)" in line and "temporale_prob" not in data:
            value = lines[i + 1].strip()
            data["temporale_prob"] = int(value) if value.isdigit() else value

        if "Quota zero termico (m)" in line and "quota_zero_termico" not in data:
            data["quota_zero_termico"] = int(lines[i + 1])

        if "Quota delle nevicate (m)" in line and "quota_nevicate" not in data:
            data["quota_nevicate"] = int(lines[i + 1])

        if "Vento medio a 2.000 m (m/s)" in line and "vento_2000_direzione" not in data:
            data["vento_2000_direzione"] = lines[i + 1].strip()
            data["vento_2000_velocita"] = float(lines[i + 2])

        if "Vento medio a 3.000 m (m/s)" in line and "vento_3000_direzione" not in data:
            data["vento_3000_direzione"] = lines[i + 1].strip()
            data["vento_3000_velocita"] = float(lines[i + 2])

        if "Tarvisio" in line and "tarvisio_min" not in data:
            data["tarvisio_min"] = int(lines[i + 1])
            data["tarvisio_max"] = int(lines[i + 2])

        if "M. Lussari" in line and "m_lussari_min" not in data:
            data["m_lussari_min"] = int(lines[i + 1])
            data["m_lussari_max"] = int(lines[i + 2])

    return data


def handler_prealpi_carniche(lines):
    data = {}

    for i, line in enumerate(lines):
        if (
            "Temperatura media a 1.000 m (°C)" in line
            and "temperatura_media_1000" not in data
        ):
            data["temperatura_media_1000"] = int(lines[i + 1])

        if (
            "Temperatura media a 2.000 m (°C)" in line
            and "temperatura_media_2000" not in data
        ):
            data["temperatura_media_2000"] = int(lines[i + 1])

        if (
            "Probabilità precipitazioni estese (%)" in line
            and "pioggia_prob" not in data
        ):
            value = lines[i + 1].strip()
            data["pioggia_prob"] = int(value) if value.isdigit() else value

        if "Probabilità di temporali (%)" in line and "temporale_prob" not in data:
            value = lines[i + 1].strip()
            data["temporale_prob"] = int(value) if value.isdigit() else value

        if "Quota zero termico (m)" in line and "quota_zero_termico" not in data:
            data["quota_zero_termico"] = int(lines[i + 1])

        if "Quota delle nevicate (m)" in line and "quota_nevicate" not in data:
            data["quota_nevicate"] = int(lines[i + 1])

        if "Vento medio a 2.000 m (m/s)" in line and "vento_2000_direzione" not in data:
            data["vento_2000_direzione"] = lines[i + 1].strip()
            data["vento_2000_velocita"] = float(lines[i + 2])

        if "Vento medio a 3.000 m (m/s)" in line and "vento_3000_direzione" not in data:
            data["vento_3000_direzione"] = lines[i + 1].strip()
            data["vento_3000_velocita"] = float(lines[i + 2])

        if "Claut" in line and "claut_min" not in data:
            data["claut_min"] = int(lines[i + 1])
            data["claut_max"] = int(lines[i + 2])

        if "Piancavallo" in line and "piancavallo_min" not in data:
            data["piancavallo_min"] = int(lines[i + 1])
            data["piancavallo_max"] = int(lines[i + 2])

    return data


def handler_prealpi_giulie(lines):
    data = {}

    for i, line in enumerate(lines):
        if (
            "Temperatura media a 1.000 m (°C)" in line
            and "temperatura_media_1000" not in data
        ):
            data["temperatura_media_1000"] = int(lines[i + 1])

        if (
            "Temperatura media a 2.000 m (°C)" in line
            and "temperatura_media_2000" not in data
        ):
            data["temperatura_media_2000"] = int(lines[i + 1])

        if (
            "Probabilità precipitazioni estese (%)" in line
            and "pioggia_prob" not in data
        ):
            value = lines[i + 1].strip()
            data["pioggia_prob"] = int(value) if value.isdigit() else value

        if "Probabilità di temporali (%)" in line and "temporale_prob" not in data:
            value = lines[i + 1].strip()
            data["temporale_prob"] = int(value) if value.isdigit() else value

        if "Quota zero termico (m)" in line and "quota_zero_termico" not in data:
            data["quota_zero_termico"] = int(lines[i + 1])

        if "Quota delle nevicate (m)" in line and "quota_nevicate" not in data:
            data["quota_nevicate"] = int(lines[i + 1])

        if "Vento medio a 2.000 m (m/s)" in line and "vento_2000_direzione" not in data:
            data["vento_2000_direzione"] = lines[i + 1].strip()
            data["vento_2000_velocita"] = float(lines[i + 2])

        if "Vento medio a 3.000 m (m/s)" in line and "vento_3000_direzione" not in data:
            data["vento_3000_direzione"] = lines[i + 1].strip()
            data["vento_3000_velocita"] = float(lines[i + 2])

        if "Sella Nevea" in line and "sella_nevea_min" not in data:
            data["sella_nevea_min"] = int(lines[i + 1])
            data["sella_nevea_max"] = int(lines[i + 2])

        if "M. Canin (R. Gilberti)" in line and "m_canin_min" not in data:
            data["m_canin_min"] = int(lines[i + 1])
            data["m_canin_max"] = int(lines[i + 2])

    return data


def handler_alta_pianura(lines):
    data = {}

    for i, line in enumerate(lines):
        if (
            "Temperatura minima pianura (°C)" in line
            and "temperatura_minima_pianura" not in data
        ):
            data["temperatura_minima_pianura"] = lines[i + 1].strip()

        if (
            "Temperatura massima pianura (°C)" in line
            and "temperatura_massima_pianura" not in data
        ):
            data["temperatura_massima_pianura"] = lines[i + 1].strip()

        if "Quota zero termico (m)" in line and "quota_zero_termico" not in data:
            data["quota_zero_termico"] = int(lines[i + 1].strip())

        if "Quota delle nevicate (m)" in line and "quota_nevicate" not in data:
            data["quota_nevicate"] = int(lines[i + 1].strip())

        if (
            "Probabilità precipitazioni estese (%)" in line
            and "pioggia_prob_prealpi" not in data
        ):
            data["pioggia_prob_prealpi"] = int(lines[i + 1].strip())
            data["pioggia_prob_pianura"] = int(lines[i + 2].strip())

        if (
            "Probabilità di temporali (%)" in line
            and "temporale_prob_prealpi" not in data
        ):
            data["temporale_prob_prealpi"] = int(lines[i + 1].strip())
            data["temporale_prob_pianura"] = int(lines[i + 2].strip())

    return data


def handler_costa(lines):
    data = {}

    for i, line in enumerate(lines):
        if "Temperatura minima (°C)" in line and "temperatura_minima" not in data:
            data["temperatura_minima"] = lines[i + 1].strip()

        if "Temperatura massima (°C)" in line and "temperatura_massima" not in data:
            data["temperatura_massima"] = lines[i + 1].strip()

        if (
            "Probabilità precipitazioni estese (%)" in line
            and "pioggia_prob" not in data
        ):
            data["pioggia_prob"] = int(lines[i + 1].strip())

        if "Probabilità di temporali (%)" in line and "temporale_prob" not in data:
            data["temporale_prob"] = int(lines[i + 1].strip())

        if (
            "Vento medio al largo: direzione ed intensità (kt)" in line
            and "vento_mattino_direzione" not in data
        ):
            # Mattino
            if i + 2 < len(lines):
                mattino = lines[i + 2].strip().split()
                if len(mattino) >= 2:
                    data["vento_mattino_direzione"] = mattino[0]
                    data["vento_mattino_intensita"] = mattino[1]

            if i + 4 < len(lines):
                pomeriggio = lines[i + 4].strip().split()
                if len(pomeriggio) >= 2:
                    data["vento_pomeriggio_direzione"] = pomeriggio[0]
                    data["vento_pomeriggio_intensita"] = pomeriggio[1]

    return data


if __name__ == "__main__":
    target_date = datetime(2019, 1, 1)
    data = text_extraction(target_date)
    print(data)
