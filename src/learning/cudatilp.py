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
    """
    FOLD-R style training: rules are learned only for `target` and every
    other row is left to the default (the model predicts None for it).

    FOLD-RM instead takes the most frequent label as head each round, so on
    a binary task it mostly describes the majority side and leaves the
    interesting class as a catch-all `month>X ; month=<X` default.
    """
    from src.algos.algo import cover, learn_rule

    pos = [d for d in data if d[-1] == target]
    neg = [d for d in data if d[-1] != target]
    rules = []
    while pos:
        rule = learn_rule(pos, neg, [], ratio)[0]
        if not rule[1]:
            break
        rest = [d for d in pos if not cover(rule, d)]
        if len(rest) == len(pos):
            break
        pos = rest
        rules.append(((-1, "==", target), rule[1], rule[2], rule[3]))
    model.rules = rules


def prune_rules(model, data: list[list], min_support: int) -> None:
    """Drops the rules that hold for fewer than `min_support` training rows
    of their own head label: they explain a handful of days and mostly fit
    noise."""
    from src.algos.algo import evaluate

    if min_support <= 0:
        return
    model.rules = [
        r
        for r in model.rules
        if sum(1 for d in data if d[-1] == r[0][2] and evaluate(r, d)) >= min_support
    ]
    model.asp_rules = None  # asp() caches the decoded rules
