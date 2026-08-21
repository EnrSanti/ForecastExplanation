import argparse
import logging
import sys

import yaml

import data_extraction
import image_processing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ForecastExplanation")


def main():
    parser = argparse.ArgumentParser(description="ForecastExplanation Pipeline")
    parser.add_argument("--config", type=str, help="Path to config YAML file containing dates", default="config.yaml")
    parser.add_argument("-c", dest="clean", action="count", default=0, help="Clean tmp folders: -cc to delete all")
    parser.add_argument("-m", "--in-memory", dest="in_memory", action="store_true",
                        help="Pass intermediate data in memory instead of writing to disk")

    args, unknown = parser.parse_known_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        dates = config.get("dates", [])

    if not dates:
        logger.error("No dates provided. ")
        sys.exit(1)

    data_extraction.extract(dates, data_extraction.Region.FVG, clean_level=args.clean, in_memory=args.in_memory)

    #data_extraction.extract(dates, data_extraction.Region.FVG, output_dir="./tmp_data/clustered", clean_level=args.clean)
    image_processing.run_tobac(dates, input_dir="./tmp_data/clustered", output_dir=image_processing.TOBAC_OUTPUT, region=image_processing.Region.FVG)

if __name__ == "__main__":
    main()
