import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("ForecastExplanation")


class BaseTranslator(ABC):
    extension: str = ""

    def translate(
        self,
        dates: list[datetime],
        input_folder: str | Path,
        output_folder: str | Path,
        force: bool = False,
    ) -> None:
        """
        Translates the reasoning files for each requested date.

        Args:
            dates: The dates to translate.
            input_folder: The root folder containing date-formatted subdirectories.
            output_folder: The root folder where translated files should be saved.
            force: If True, re-translates a day even if its output file already exists.
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        ext = self.extension.strip(".") or "txt"

        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            item = input_path / date_str

            if not item.is_dir():
                logger.warning(
                    f"No data found for {date_str} in {input_path}. Skipping translation."
                )
                continue

            output_file = output_path / f"{date_str}.{ext}"

            if not force and output_file.exists():
                continue

            result_bytes = self.translate_day(item)
            output_file.write_bytes(result_bytes)

    @abstractmethod
    def translate_day(self, day_input_folder: Path) -> bytes:
        """
        Translates the reasoning files for a specific day.

        Args:
            day_input_folder: The input folder for a specific day.

        Returns:
            bytes: The translated content to be saved to a file.
        """
