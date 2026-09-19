import datetime as dt
import json
import logging
import os
from datetime import datetime

from .extract_pdf import text_extract
from .extract_xml import xml_extract

logger = logging.getLogger("ForecastExplanation")


def generate_gt(target_dates: list[datetime], output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    for date in target_dates:
        try:
            if os.path.exists("./xmls"):
                res = xml_extract(date - dt.timedelta(days=1))
            else:
                logger.warning(
                    "XML files not found. Falling back to pdf text extraction."
                )
                res = text_extract(date - dt.timedelta(days=1))
            save_to_file(res, output_path, date)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error generating ground truth for date {date}: {e}")


def save_to_file(data: dict, output_path: str, date) -> None:
    out = os.path.join(output_path, date.strftime("%Y-%m-%d"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "gt.json"), "w") as file:
        file.write(json.dumps(data, indent=4))
