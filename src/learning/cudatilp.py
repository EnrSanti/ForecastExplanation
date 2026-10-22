import sys
from pathlib import Path

CUDATILP_ROOT = Path(__file__).resolve().parent / "CUDatILP"

if str(CUDATILP_ROOT) not in sys.path:
    sys.path.append(str(CUDATILP_ROOT))


def load_classifier_cls():
    if not (CUDATILP_ROOT / "src" / "common" / "foldrm.py").exists():
        raise RuntimeError(
            f"CUDatILP submodule not found at {CUDATILP_ROOT}. "
            "Run `git submodule update --init`."
        )
    try:
        from src.common.foldrm import Classifier
    except ImportError as e:
        raise RuntimeError(f"Could not import CUDatILP Classifier ({e})") from e
    return Classifier


def save_model(model, path: Path) -> None:
    from src.common.foldrm import save_model_to_file

    save_model_to_file(model, str(path))


def scores(Y_hat: list, Y: list, weighted: bool = False):
    from src.common.utils import scores as _scores

    return _scores(Y_hat, Y, weighted=weighted)


def fit_target(model, data: list[list], target: str, ratio: float) -> None:
    """FOLD-R style: rules only for `target`, other rows predict None."""
    from src.algos.algo import cover, learn_rule

    pos = [d for d in data if d[-1] == target]
    neg = [d for d in data if d[-1] != target]
    rules = []
    while pos:
        rule = learn_rule(pos, neg, [], ratio)
        if not rule[1]:
            break
        rest = [d for d in pos if not cover(rule, d)]
        if len(rest) == len(pos):
            break
        pos = rest
        rules.append(((-1, "==", target), rule[1], rule[2], rule[3]))
    model.rules = rules
