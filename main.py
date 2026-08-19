import argparse
import logging
import sys

import yaml

import raw_data


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ForecastExplanation")


def main():
    parser = argparse.ArgumentParser(description="ForecastExplanation Pipeline")
    parser.add_argument("--config", type=str, help="Path to config YAML file containing dates", default="config.yaml")

    args, unknown = parser.parse_known_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        dates = config.get("dates", [])

    if not dates:
        logger.error("No dates provided. ")
        sys.exit(1)

    print("-------------- From GRIB to images (V2) --------------")
    print("[1]: CUT Girb & extract DATA")
    mode = int(input("Enter mode: ").strip())

    if mode == 1:
        raw_data.extract(dates, raw_data.Region.FVG, output_dir="./raw_data/data/test")
    else:
        logger.error("Invalid option.")


if __name__ == "__main__":
    main()
