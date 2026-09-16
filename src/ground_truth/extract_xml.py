"""
Parser for meteo.fvg XML weather bulletins ("previsioni.php").
Returns a dictionary shaped like::
    {

      "Alpi Carniche": {
          "CIELO_DESCRIZIONE": "coperto",
          "TEMPORALE_DESCRIZIONE": False,
          "PIOGGIA_DESCRIZIONE": "abbondante",
      },
      "Alpi Giulie": {...},
      ...
      "Trieste": {...},
    }
"""

import html
import logging
import xml.etree.ElementTree as ET
from datetime import datetime

logger = logging.getLogger("ForecastExplanation")

A_ZONES = [f"A{i}" for i in range(1, 10)]
_NONE_VALUES = {"100", "", None}
KEYS = ["CIELO_DESCRIZIONE", "TEMPORALE_DESCRIZIONE", "PIOGGIA_DESCRIZIONE"]


def _text(el: ET.Element | None) -> str | None:
    if el is None or el.text is None:
        return None
    t = html.unescape(html.unescape(el.text)).strip()
    return t or None


def _element_to_dict(el: ET.Element) -> dict:
    return {child.tag.lower(): _text(child) for child in el}


def _iso_date(valid_date: str | None) -> str | None:
    """Converts 'DD-MM-YYYY' (the bulletin's format) to 'YYYY-MM-DD'."""
    if not valid_date:
        return None
    day, month, year = valid_date.split("-")
    return f"{year}-{month}-{day}"


def _zone_name(zone_raw: dict) -> str:
    description = zone_raw.get("descrizione") or ""
    return description.replace("_", " ").strip()


def xml_extract(date: datetime) -> dict:
    xml_path = f"xml/PW{date.strftime('%Y%m%d')}.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    deadline = root.find("previsioni/scadenze/scadenza[@id='1']")
    iso_date = _iso_date(deadline.get("data_validita"))
    a_zones = {}
    for zone_el in deadline.findall("zone/zona"):
        name = zone_el.get("nome")

        if not name in A_ZONES:
            continue

        data = _element_to_dict(zone_el)
        data["descrizione"] = zone_el.get("descrizione")

        a_zones[_zone_name(data)] = {k: data.get(k.lower()) or False for k in KEYS}

    if len(a_zones) != 9:
        logger.warning(f"Expected 9 zones, found {len(a_zones)}")

    return {iso_date: a_zones}
