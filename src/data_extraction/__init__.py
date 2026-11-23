from pathlib import Path

# GRIBs shared by every run, in the repo; the others are under the run's output_path
RAW_DATA_DIR: Path = Path(__file__).resolve().parents[2] / "tmp_data/original_CERRA"
CUT_DATA_DIR: str = "tmp_data/CERRA_cut"
DISCRETE_DATA_DIR: str = "tmp_data/extracted_data"
CLUSTERED_DATA_DIR: str = "tmp_data/clustered_data"

FRONT_LEVELS = [850, 700, 500]

from .main import clean_artifacts, extract

__all__: list[str] = [
    "CLUSTERED_DATA_DIR",
    "CUT_DATA_DIR",
    "DISCRETE_DATA_DIR",
    "RAW_DATA_DIR",
    "clean_artifacts",
    "extract",
]
