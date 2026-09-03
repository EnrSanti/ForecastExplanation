import argparse
import logging
import sys
import os
import yaml

import data_extraction

import image_processing

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("ForecastExplanation")


def parse_args_and_config():
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
        help="Force the extraction even if the data is already present, more f for more redone steps",
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

    args, unknown = parser.parse_known_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    return args, config


def main():
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

        dates = run_config.get("dates", [])
        region_str = run_config.get("region", "FVG").upper()

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

        if debug:
            logger.setLevel(logging.DEBUG)
            logging.getLogger("data_extraction").setLevel(logging.DEBUG)
            logging.getLogger("image_processing").setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            logging.getLogger("data_extraction").setLevel(logging.INFO)
            logging.getLogger("image_processing").setLevel(logging.INFO)

        if not dates:
            logger.error(f"No dates provided for {run_name}.")
            continue

        try:
            data_region = data_extraction.Region[region_str]
            img_region = image_processing.Region[region_str]
        except KeyError:
            logger.error(
                f"Invalid region '{region_str}' in {run_name}. Must be one of {[r.name for r in data_extraction.Region]}"
            )
            continue

        data_extraction.DISCRETE_DATA_DIR = os.path.join(
            "runs", run_name, data_extraction.DISCRETE_DATA_DIR
        )
        data_extraction.CLUSTERED_DATA_DIR = os.path.join(
            "runs", run_name, data_extraction.CLUSTERED_DATA_DIR
        )
        data_extraction.CUT_DATA_DIR = os.path.join(
            "runs", run_name, data_extraction.CUT_DATA_DIR
        )
        image_processing.TOBAC_OUTPUT = os.path.join(
            "runs", run_name, image_processing.TOBAC_OUTPUT
        )

        data_extraction.extract(
            dates,
            data_region,
            clean_level=clean,
            clustering=clustering,
            force_redo=force,
            just_cut=just_cut,
            create_images=save_images,
        )
        if just_cut:
            continue

        input_dir = (
            data_extraction.CLUSTERED_DATA_DIR
            if clustering
            else data_extraction.DISCRETE_DATA_DIR
        )
        image_processing.run_tobac(
            dates,
            input_dir=input_dir,
            output_dir=image_processing.TOBAC_OUTPUT,
            region=img_region,
            save_images=save_images,
        )

        logger.info(f"--- Finished {run_name} ---")


if __name__ == "__main__":
    main()
