import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from tqdm import tqdm

from region import Region

logger = logging.getLogger("ForecastExplanation")


class BaseTranslator(ABC):
    extension: str = ""
    logger = logger

    def translate(
        self,
        dates: list[date],
        input_folder: str | Path,
        output_folder: str | Path,
        region: Region,
        force: bool = False,
        workers: int = 12,
    ) -> Path | None:
        """
        Translates the reasoning files for each requested date.

        Args:
            dates: The dates to translate.
            input_folder: The root folder containing date-formatted subdirectories.
            output_folder: The root folder where translated files should be saved.
            region: The region the run was configured with (and its resolved cities).
            force: If True, re-translates a day even if its output file already exists.
            workers: Number of parallel worker processes.

        Returns:
            The path of the merged dataset, or None if nothing was merged.
        """
        logger.info("Starting translation")
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._translate_single_day,
                    target_date,
                    input_path,
                    output_path,
                    region,
                    force=force,
                ): target_date
                for target_date in dates
            }

            for future in tqdm(
                as_completed(futures), total=len(dates), desc="Translation"
            ):
                target_date = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception(f"Translation failed for {target_date}")

        merged = self.merge_into_dataset(dates, output_path)
        logger.info("Translation completed.")
        return merged

    def _translate_single_day(
        self,
        target_date: date,
        input_path: Path,
        output_path: Path,
        region: Region,
        force: bool = False,
    ) -> None:
        date_str = target_date.strftime("%Y-%m-%d")
        item = input_path / date_str

        if not item.is_dir():
            logger.warning(
                f"No data found for {date_str} in {input_path}. Skipping translation."
            )
            return

        ext = self.extension.strip(".") or "txt"
        output_file = output_path / f"{date_str}.{ext}"

        if not force and output_file.exists():
            return

        result_bytes = self.translate_day(item, target_date, region)
        output_file.write_bytes(result_bytes)

    @abstractmethod
    def translate_day(
        self, day_input_folder: Path, target_date: date, region: Region
    ) -> bytes:
        """
        Translates the reasoning files for a specific day.

        Args:
            day_input_folder: The input folder for a specific day.
            target_date: The date for which to translate the data.
            region: The region the run was configured with (and its resolved cities).

        Returns:
            bytes: The translated content to be saved to a file.
        """

    @abstractmethod
    def merge_into_dataset(self, dates: list[date], input_folder: Path) -> Path | None:
        """
        Merges the per-day translated files into a single combined dataset file.

        Args:
            dates: The dates whose translated files should be merged.
            input_folder: The folder containing the per-day translated files.

        Returns:
            The path of the merged dataset, or None if nothing was merged.
        """
