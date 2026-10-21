import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger("ForecastExplanation")


class BaseTranslator(ABC):
    extension: str  # output file extension, without the dot
    logger = logger

    def translate(
        self,
        dates: list[date],
        input_folder: Path,
        output_folder: Path,
        force: bool = False,
        workers: int = 12,
    ) -> tuple[Path | None, list[date]]:
        """
        Translates the reasoning files for each requested date.

        Args:
            dates: The dates to translate.
            input_folder: The root folder containing date-formatted subdirectories.
            output_folder: The root folder where translated files should be saved.
            force: If True, re-translates a day even if its output file already exists.
            workers: Number of parallel worker processes.

        Returns:
            The path of the merged dataset (None if nothing was merged) and the
            days translated without errors, the only ones merged into it.
        """
        logger.info("Starting translation")
        output_folder.mkdir(parents=True, exist_ok=True)

        ok = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._translate_single_day,
                    target_date,
                    input_folder,
                    output_folder,
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
                    ok.append(target_date)
                except Exception:
                    logger.exception(f"Translation failed for {target_date}")

        ok.sort()
        merged = self.merge_into_dataset(ok, output_folder)
        logger.info("Translation completed.")
        return merged, ok

    def _translate_single_day(
        self,
        target_date: date,
        input_path: Path,
        output_path: Path,
        force: bool = False,
    ) -> None:
        date_str = target_date.strftime("%Y-%m-%d")
        item = input_path / date_str

        if not item.is_dir():
            raise FileNotFoundError(f"No data found for {date_str} in {input_path}")

        output_file = output_path / f"{date_str}.{self.extension}"

        if not force and output_file.exists():
            return

        result_bytes = self.translate_day(item, target_date)
        output_file.write_bytes(result_bytes)

    @abstractmethod
    def translate_day(self, day_input_folder: Path, target_date: date) -> bytes:
        """
        Translates the reasoning files for a specific day.

        Args:
            day_input_folder: The input folder for a specific day.
            target_date: The date for which to translate the data.

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
