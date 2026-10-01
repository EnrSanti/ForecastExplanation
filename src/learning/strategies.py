"""
How a multi-class target is framed for FOLD-RM.

FOLD-RM is a one-vs-rest covering loop that takes the most frequent label as
positive each round. On noisy, imbalanced multi-class targets (prev_cloud has
5 classes, majority ~28%) `gain()` rejects nearly every literal and the loop
stops after a couple of rules, leaving most test rows uncovered. Training one
binary model per class works much better, so OneVsRest is the default.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

Row = list
Task = tuple[str, list[Row], list[Row]]


def majority_label(data: list[Row]) -> str:
    labels = [d[-1] for d in data]
    return max(set(labels), key=labels.count)


class Strategy(ABC):
    @abstractmethod
    def tasks(self, train: list[Row], test: list[Row]) -> Iterable[Task]:
        """Yields (task_name, train, test) triples, one per model to fit."""


class OneVsRest(Strategy):
    @staticmethod
    def binarize(data: list[Row], positive: str) -> list[Row]:
        """Relabels each row as 'pos' if its label equals `positive`, else 'neg'."""
        return [d[:-1] + ["pos" if d[-1] == positive else "neg"] for d in data]

    def tasks(self, train: list[Row], test: list[Row]) -> Iterable[Task]:
        for positive in sorted({d[-1] for d in train}):
            yield (
                f"class_{positive}_vs_rest",
                self.binarize(train, positive),
                self.binarize(test, positive),
            )


class Multiclass(Strategy):
    def tasks(self, train: list[Row], test: list[Row]) -> Iterable[Task]:
        yield "multiclass", train, test


STRATEGIES: dict[str, type[Strategy]] = {
    "one_vs_rest": OneVsRest,
    "multiclass": Multiclass,
}
