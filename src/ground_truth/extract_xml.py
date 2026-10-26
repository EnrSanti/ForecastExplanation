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
from datetime import date
from pathlib import Path

logger = logging.getLogger("ForecastExplanation")
XML_DIR = Path(__file__).resolve().parents[2] / "xmls"

A_ZONES = [f"A{i}" for i in range(1, 10)]
KEYS = ["CIELO_DESCRIZIONE", "TEMPORALE_DESCRIZIONE", "PIOGGIA_DESCRIZIONE"]


def _text(el: ET.Element | None) -> str | None:
    if el is None or el.text is None:
        return None
    t = html.unescape(html.unescape(el.text)).strip()
    return t or None


def xml_extract(target_date: date) -> dict:
    root = ET.parse(XML_DIR / f"PW{target_date:%Y%m%d}.xml").getroot()
    deadline = root.find("previsioni/scadenze/scadenza[@id='1']")
    a_zones = {}
    for zone_el in deadline.findall("zone/zona"):
        if zone_el.get("nome") not in A_ZONES:
            continue
        name = (zone_el.get("descrizione") or "").replace("_", " ").strip()
        if not name:
            logger.warning(
                f"{target_date}: zone {zone_el.get('nome')} has no name, skipped"
            )
            continue
        a_zones[name] = {k: _text(zone_el.find(k)) or False for k in KEYS}

    if len(a_zones) != 9:
        logger.warning(f"Expected 9 zones, found {len(a_zones)}")

    return a_zones
