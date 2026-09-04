import logging
import os
import xarray as xr
from tqdm import tqdm

from region import Region
from .wind import detect_winds
from .heat import detect_heat
from .utils import get_heights

logger = logging.getLogger("ForecastExplanation")


def reason(
    dates: list, input_dir: str, output_dir: str, region: Region, force: bool = False
):
    """
    Perform reasoning on the input data (nc format) and save the results to the output path (text format).
    Converts raw data to reasoning data, ready to be converted to ASP formats.

    Args:
        dates (list): List of dates for which reasoning is to be performed.
        input_dir (str): Path to the input directory containing spatial data.
        output_dir (str): Path to the directory where processed output will be written.
        region (Region): The specific geographic region to be used.
        force (bool, optional): If True, forces the processing of all dates. Defaults to False.
    """
    logger.info("Starting reasoning")
    for date in tqdm(dates, desc="Reasoning"):
        day_input_dir = os.path.join(input_dir, date.strftime("%Y-%m-%d"))
        day_output_dir = os.path.join(
            output_dir, date.strftime("%Y-%m-%d"), "reasoning"
        )

        if not force and os.path.exists(day_output_dir) and os.listdir(day_output_dir):
            logger.debug(
                f"Reasoning already exists for {date.strftime('%Y-%m-%d')}. Skipping."
            )
            continue

        os.makedirs(day_output_dir, exist_ok=True)
        logger.debug(f"Processing reasoning for {date.strftime('%Y-%m-%d')}")

        with xr.open_dataset(os.path.join(day_input_dir, "segmentation.nc")) as seg_ds:
            pass # Keep opening if needed later, but heights comes from feat_ds

        with xr.open_dataset(os.path.join(day_input_dir, "features.nc")) as feat_ds:
            heights = get_heights(feat_ds)
            detect_winds(
                feat_ds,
                region.get_cities(),
                os.path.join(day_output_dir, "winds.txt"),
                heights,
            )

            detect_heat(
                feat_ds,
                region.get_cities(),
                os.path.join(day_output_dir, "heat.txt"),
                heights,
            )
