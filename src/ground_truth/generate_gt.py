import json
import logging
from datetime import date, timedelta
from pathlib import Path

from .extract_xml import xml_extract

logger = logging.getLogger("ForecastExplanation")


def generate_gt(
    target_dates: list[date], output_path: Path, force: bool = False
) -> list[date]:
    ok = []
    for target_date in target_dates:
        day = target_date.strftime("%Y-%m-%d")
        gt_path = output_path / day / "gt.json"
        if not force and gt_path.exists():
            logger.debug(f"Ground truth already exists for {day}. Skipping generation.")
            ok.append(target_date)
            continue
        try:
            gt = xml_extract(target_date - timedelta(days=1))
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.write_text(json.dumps(gt, indent=4))
            ok.append(target_date)
        except Exception:
            logger.exception(f"Error generating ground truth for date {target_date}")

    return ok
