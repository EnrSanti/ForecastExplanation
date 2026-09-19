import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class BaseTranslator(ABC):
    extension: str = ""

    def translate(
        self, 
        input_folder: Union[str, Path], 
        output_folder: Union[str, Path]
    ) -> None:
        """
        Translates the reasoning files for each day in the input folder.

        Args:
            input_folder: The root folder containing date-formatted subdirectories.
            output_folder: The root folder where translated files should be saved.
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for item in input_path.iterdir():
            if item.is_dir() and date_pattern.match(item.name):
                result_bytes = self.translate_day(item)
                ext = self.extension.strip(".")

                if not ext:
                    ext = "txt"

                output_file = output_path / f"{item.name}.{ext}"
                output_file.write_bytes(result_bytes)

    @abstractmethod
    def translate_day(
        self, 
        day_input_folder: Path
    ) -> bytes:
        """
        Translates the reasoning files for a specific day.

        Args:
            day_input_folder: The input folder for a specific day.
            
        Returns:
            bytes: The translated content to be saved to a file.
        """
        pass
