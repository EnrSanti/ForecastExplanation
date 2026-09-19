import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv

import data_extraction
import features_detection
import ground_truth
import reasoning
import translators
from region import Region

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("ForecastExplanation")


def parse_args_and_config() -> tuple[argparse.Namespace, dict]:
    load_dotenv()
    parser = argparse.ArgumentParser(description="ForecastExplanation Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file containing dates",
        default="config.yaml",
    )
    parser.add_argument(
        "-c",
        dest="clean",
        action="count",
        default=0,
        help="Clean tmp folders: -cc to delete all",
    )
    parser.add_argument(
        "-f",
        dest="force",
        action="count",
        default=0,
        help="-f forces reasoning, -ff forces feature extraction, -fff forces data extraction",
    )
    parser.add_argument(
        "--clustering", action="store_true", help="Toggle clustering in data extraction"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging for the application",
    )
    parser.add_argument(
        "-jc",
        "--just-cut",
        action="store_true",
        help="Just download and cut the GRIB files, skipping feature extraction and clustering, no images generated",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Generate visualization images of the tracking results",
    )

    args, _ = parser.parse_known_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    return args, config


def _parse_date_value(value: datetime | date | str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        d = date.fromisoformat(value)
        return datetime(d.year, d.month, d.day)

    raise ValueError(f"Unsupported date value: {value!r}")


def parse_dates(dates_entry: list | str | dict | None) -> list[datetime]:
    """
    Parses a date configuration entry, which can be a single date item or a list of items.
    """

    if not dates_entry:
        return []

    if not isinstance(dates_entry, list):
        dates_entry = [dates_entry]

    parsed_dates = set()

    for item in dates_entry:
        if isinstance(item, (datetime, date, str)):
            parsed_dates.add(_parse_date_value(item))
        elif isinstance(item, dict):
            start = _parse_date_value(item.get("start"))
            end = _parse_date_value(item.get("end"))

            curr = start
            step = item.get("step", 1)
            while curr <= end:
                parsed_dates.add(curr)
                curr += timedelta(days=step)

    return sorted(parsed_dates)


def main() -> None:
    args, config = parse_args_and_config()

    if "dates" in config or "region" in config:
        runs = {"default_run": config}
    else:
        runs = config

    if not runs:
        logger.error("No runs found in config.")
        sys.exit(1)

    for run_name, run_config in runs.items():
        logger.info(f" --- Starting {run_name} ---")

        raw_dates = run_config.get("dates", [])
        dates = parse_dates(raw_dates)

        clean = args.clean if args.clean else run_config.get("clean", 0)
        force = args.force if args.force else run_config.get("force", 0)
        clustering = (
            args.clustering if args.clustering else run_config.get("clustering", False)
        )
        debug = args.debug if args.debug else run_config.get("debug", False)
        just_cut = args.just_cut if args.just_cut else run_config.get("just_cut", False)
        save_images = (
            args.save_images
            if args.save_images
            else run_config.get("save_images", False)
        )
        output_path = Path(run_config.get("output_path", Path("runs") / run_name))

        if debug:
            logger.setLevel(logging.DEBUG)
            logging.getLogger("data_extraction").setLevel(logging.DEBUG)
            logging.getLogger("features_detection").setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            logging.getLogger("data_extraction").setLevel(logging.INFO)
            logging.getLogger("features_detection").setLevel(logging.INFO)

        if not dates:
            logger.error(f"No dates provided for {run_name}.")
            continue

        try:
            region = Region.from_config(
                run_config.get("region", "FVG"), cities=run_config.get("cities", None)
            )
        except ValueError as e:
            logger.error(f"Region error in {run_name}: {e}")
            continue

        data_extraction.extract(
            dates,
            region,
            output_path=output_path,
            clean_level=clean,
            clustering=clustering,
            force_redo=force > 3,
            just_cut=just_cut,
            create_images=save_images,
        )
        if just_cut:
            logger.info(f"{run_name} finished just cut.")
            continue

        input_dir = (
            output_path / data_extraction.CLUSTERED_DATA_DIR
            if clustering
            else output_path / data_extraction.DISCRETE_DATA_DIR
        )
        features_detection.run_tobac(
            dates,
            input_dir=input_dir,
            output_dir=output_path,
            region=region,
            force=force > 2,
            save_images=save_images,
        )
        reasoning.reason(dates, output_path, output_path, region, force=force > 1)
        ground_truth.generate_gt(dates, output_path, force=force > 1)

        translate_output_path = os.path.join(output_path, "translated")
        translators.FoldRmTranslator().translate(
            output_path, translate_output_path, force=force > 0
        )

        logger.info(f"--- Finished {run_name} ---\n\n")


if __name__ == "__main__":
    main()
