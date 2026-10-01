import sys
from pathlib import Path

CUDATILP_ROOT = Path(__file__).resolve().parent / "CUDatILP"

if str(CUDATILP_ROOT) not in sys.path:
    sys.path.insert(0, str(CUDATILP_ROOT))


def load_classifier_cls():
    if not (CUDATILP_ROOT / "src" / "common" / "foldrm.py").exists():
        raise RuntimeError(
            f"CUDatILP submodule not found at {CUDATILP_ROOT}. "
            "Run `git submodule update --init`."
        )
    try:
        from src.common.foldrm import Classifier
    except ImportError as e:
        raise RuntimeError(
            f"Could not import CUDatILP Classifier ({e}). "
            "Run `git submodule update --init`."
        ) from e
    return Classifier


def save_model(model, path: Path) -> None:
    from src.common.foldrm import save_model_to_file

    save_model_to_file(model, str(path))


def scores(Y_hat: list, Y: list, weighted: bool = False):
    from src.common.utils import scores as _scores

    return _scores(Y_hat, Y, weighted=weighted)
