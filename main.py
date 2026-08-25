import argparse
import logging
import sys

import yaml

import data_extraction

import image_processing

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger("ForecastExplanation")


def main():
    parser = argparse.ArgumentParser(description="ForecastExplanation Pipeline")
    parser.add_argument("--config", type=str, help="Path to config YAML file containing dates", default="config.yaml")
    parser.add_argument("-c", dest="clean", action="count", default=0, help="Clean tmp folders: -cc to delete all")
    parser.add_argument("-f", dest="force", action="count", default=0,
                        help="Force the extraction even if the data is already present, more f for more redone steps")
    parser.add_argument("-m", "--in-memory", dest="in_memory", action="store_true",
                        help="Pass intermediate data in memory instead of writing to disk")
    parser.add_argument("--clustering", action="store_true", help="Toggle clustering in data extraction")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging for the application")

    args, unknown = parser.parse_known_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("data_extraction").setLevel(logging.DEBUG)
        logging.getLogger("image_processing").setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        logging.getLogger("data_extraction").setLevel(logging.INFO)
        logging.getLogger("image_processing").setLevel(logging.INFO)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        dates = config.get("dates", [])

    if not dates:
        logger.error("No dates provided. ")
        sys.exit(1)

    data_extraction.extract(dates, data_extraction.Region.FVG, clean_level=args.clean, in_memory=args.in_memory, clustering=args.clustering, force_redo=args.force)
    input_dir = data_extraction.CLUSTERED_DATA_DIR if args.clustering else data_extraction.DISCRETE_DATA_DIR
    image_processing.run_tobac(dates, input_dir=input_dir, output_dir=image_processing.TOBAC_OUTPUT, region=image_processing.Region.FVG)


if __name__ == "__main__":
    main()
