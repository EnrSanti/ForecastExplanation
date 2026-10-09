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
        rule = learn_rule(pos, neg, [], ratio)[0]
        if not rule[1]:
            break
        rest = [d for d in pos if not cover(rule, d)]
        if len(rest) == len(pos):
            break
        pos = rest
        rules.append(((-1, "==", target), rule[1], rule[2], rule[3]))
    model.rules = rules


def _prune_exceptions(rule, rows, min_support, label, asserts_label=True):
    from src.algos.algo import evaluate

    head, items, ab, flag = rule
    if not ab:
        return rule
    body = [d for d in rows if all(evaluate(i, d) for i in items)]
    kept = []
    for exception in ab:
        exception = _prune_exceptions(
            exception, body, min_support, label, not asserts_label
        )
        fixed = sum(
            1
            for d in body
            if evaluate(exception, d) and (d[-1] == label) != asserts_label
        )
        if fixed >= min_support:
            kept.append(exception)
    return head, items, kept, flag


def prune_rules(
    model, data: list[list], min_support: int, min_exception_support: int = 0
) -> None:
    from src.algos.algo import evaluate

    if min_exception_support > 0:
        model.rules = [
            _prune_exceptions(r, data, min_exception_support, r[0][2])
            for r in model.rules
        ]
    if min_support > 0:
        model.rules = [
            r
            for r in model.rules
            if sum(1 for d in data if d[-1] == r[0][2] and evaluate(r, d))
            >= min_support
        ]
    model.asp_rules = None  # asp() cache
