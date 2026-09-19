import datetime as dt
import json
import logging
from datetime import datetime
from pathlib import Path

from .extract_pdf import text_extract
from .extract_xml import xml_extract

logger = logging.getLogger("ForecastExplanation")


def generate_gt(
    target_dates: list[datetime], output_path: Path, force: bool = False
) -> None:
    for date in target_dates:
        output_path.mkdir(parents=True, exist_ok=True)
        if not force and (output_path / date.strftime("%Y-%m-%d") / "gt.json").exists():
            logger.info(
                f"Ground truth already exists in {output_path} for {date.strftime('%Y-%m-%d')}. Skipping generation."
            )
            continue
        try:
            if Path("./xmls").exists() and date >= datetime(2021, 1, 2):  # todo
                res = xml_extract(date - dt.timedelta(days=1))
            else:
                logger.warning(
                    "XML files not found. Falling back to pdf text extraction."
                )
                res = text_extract(date - dt.timedelta(days=1))
            save_to_file(res, output_path, date)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error generating ground truth for date {date}: {e}")


def save_to_file(data: dict, output_path: Path, date) -> None:
    out = output_path / date.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gt.json", "w") as file:
        file.write(json.dumps(data, indent=4))
