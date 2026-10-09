"""
How a multi-class target is framed for FOLD-RM.

FOLD-RM is a one-vs-rest covering loop that takes the most frequent label as
positive each round. On noisy, imbalanced multi-class targets (prev_cloud has
5 classes, majority ~28%) `gain()` rejects nearly every literal and the loop
stops after a couple of rules, leaving most test rows uncovered. Training one
binary model per class works much better, an ordinal chain of
"label >= class" models better still (both targets are ordered scales).
"""

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable

Row = list
Task = tuple[str, list[Row], list[Row]]


def majority_label(data: list[Row]) -> str:
    labels = [d[-1] for d in data]
    return max(sorted(set(labels)), key=labels.count)


class Strategy(ABC):
    def target_of(self, task: str) -> str | None:
        """Label the rules of `task` are learned for, None = FOLD-RM picks."""
        return None

    @abstractmethod
    def tasks(self, train: list[Row], test: list[Row]) -> Iterable[Task]:
        """Yields (task_name, train, test) triples, one per model to fit."""

    @abstractmethod
    def combine(
        self, predictions: dict[str, list], train: list[Row], n: int
    ) -> list[str]:
        """Merges the tasks' predictions (None = no rule fired) into one
        original class per row."""


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

    def combine(
        self, predictions: dict[str, list], train: list[Row], n: int
    ) -> list[str]:
        prior = Counter(d[-1] for d in train)
        default = majority_label(train)
        votes = {
            t.removeprefix("class_").removesuffix("_vs_rest"): p
            for t, p in predictions.items()
        }
        out = []
        for i in range(n):
            pos = [c for c, p in votes.items() if p[i] == "pos"]
            out.append(max(pos, key=lambda c: prior[c]) if pos else default)
        return out


class Multiclass(Strategy):
    def tasks(self, train: list[Row], test: list[Row]) -> Iterable[Task]:
        yield "multiclass", train, test

    def combine(
        self, predictions: dict[str, list], train: list[Row], n: int
    ) -> list[str]:
        default = majority_label(train)
        return [default if p is None else p for p in predictions["multiclass"]]


class Ordinal(Strategy):
    """`ge_<c>` models "label >= c"; a row takes the class below the first
    model that doesn't fire."""

    def target_of(self, task: str) -> str:
        return task

    def tasks(self, train: list[Row], test: list[Row]) -> Iterable[Task]:
        self.classes = sorted({d[-1] for d in train}, key=float)
        for k, c in enumerate(self.classes[1:], start=1):
            up = set(self.classes[k:])

            def relabel(data, c=c, up=up):
                return [
                    d[:-1] + [f"ge_{c}" if d[-1] in up else f"lt_{c}"] for d in data
                ]

            yield f"ge_{c}", relabel(train), relabel(test)

    def combine(
        self, predictions: dict[str, list], train: list[Row], n: int
    ) -> list[str]:
        chain = [predictions[f"ge_{c}"] for c in self.classes[1:]]
        out = []
        for i in range(n):
            level = 0
            while level < len(chain) and chain[level][i] is not None:
                level += 1
            out.append(self.classes[level])
        return out


STRATEGIES: dict[str, type[Strategy]] = {
    "one_vs_rest": OneVsRest,
    "multiclass": Multiclass,
    "ordinal": Ordinal,
}
