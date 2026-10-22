import numpy as np

from .cleaner import BaseCleaner
from .mdl import Data, mdl_prune, snap_thresholds, to_node, to_rule


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


class FoldRmCleaner(BaseCleaner):
    """
    Cleaner of CUDatILP FOLD-RM rules.

    Args:
        min_support: Drop the rules holding for fewer training rows of their
            own label (0 keeps every rule).
        min_exception_support: Drop the exceptions correcting fewer training
            rows (0 keeps every exception).
        mdl_lambda: Delete rules, exceptions, atoms and interval bounds while
            training errors + lambda * conditions does not grow, lambda being
            mdl_lambda per 1000 rows of the model's minority side (so rare
            classes are not pruned away). None skips it.
        snap: Move each numeric threshold to the round value of its unit
            (5 %, 1 K, 1 h, 5000 km2, ...) with the fewest training errors.

    mdl_lambda and snap need FOLD-R style models, whose rules all have the
    same head label (the ordinal strategies).
    """

    def __init__(
        self,
        min_support: int = 0,
        min_exception_support: int = 0,
        mdl_lambda: float | None = None,
        snap: bool = False,
    ):
        self.min_support = min_support
        self.min_exception_support = min_exception_support
        self.mdl_lambda = mdl_lambda
        self.snap = snap

    def clean(self, model, data: list[list]) -> None:
        prune_rules(model, data, self.min_support, self.min_exception_support)
        if not model.rules or (self.mdl_lambda is None and not self.snap):
            return

        heads = {r[0][2] for r in model.rules}
        if len(heads) > 1:
            raise ValueError(
                f"mdl_lambda / snap need one head label per model (ordinal "
                f"strategies), got {sorted(heads)}"
            )
        positive = heads.pop()
        rows = Data(data)
        y = np.array([v == positive for v in rows.y])
        nodes = [to_node(r) for r in model.rules]
        if self.mdl_lambda is not None:
            n_pos = int(y.sum())
            lam = float(self.mdl_lambda) * min(n_pos, rows.n - n_pos) / 1000.0
            nodes = mdl_prune(nodes, rows, y, lam)
        if self.snap:
            nodes = snap_thresholds(nodes, rows, y, model.attrs)
        model.rules = [to_rule(n) for n in nodes]
