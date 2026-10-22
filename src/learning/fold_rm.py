import csv
import itertools
import json
import logging
import random
import warnings
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from timeit import default_timer as timer

from sklearn.model_selection import StratifiedGroupKFold

from cleaning import BaseCleaner, FoldRmCleaner
from utils import DATE_COLUMN

from .cudatilp import load_classifier_cls, save_model, scores
from .strategies import STRATEGIES, majority_label

logger = logging.getLogger("ForecastExplanation")

METRICS_FILE = "metrics.json"


def stratified_split(
    data: list[list], test_ratio: float, seed: int
) -> tuple[list[list], list[list]]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in data:
        by_label[row[-1]].append(row)

    train, test = [], []
    for label in sorted(by_label):
        rows = by_label[label]
        rng.shuffle(rows)
        n_test = round(len(rows) * test_ratio)
        test += rows[:n_test]
        train += rows[n_test:]
    return train, test


def date_split(
    data: list[list], dates: list[str], test_ratio: float, seed: int
) -> tuple[list[list], list[list]]:
    """Holds out whole days, so rows of the same day (one per location, sharing
    the same weather situation) never end up on both sides of the split."""
    days = sorted(set(dates))
    random.Random(seed).shuffle(days)
    test_days = set(days[: round(len(days) * test_ratio)])

    train, test = [], []
    for row, day in zip(data, dates):
        (test if day in test_days else train).append(row)
    return train, test


def homogeneous_split(
    data: list[list], dates: list[str], test_ratio: float, seed: int
) -> tuple[list[list], list[list]]:
    """Holds out whole days like date_split, choosing them so that train and
    test keep (best effort) the label proportions of the whole dataset"""
    fraction = Fraction(test_ratio).limit_denominator(20)
    fraction = min(max(fraction, Fraction(1, 20)), Fraction(19, 20))
    folds = StratifiedGroupKFold(
        n_splits=fraction.denominator, shuffle=True, random_state=seed
    )
    labels = [row[-1] for row in data]
    try:
        # sklearn warns catch
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            in_test = {
                i
                for _, test_idx in itertools.islice(
                    folds.split(data, labels, groups=dates), fraction.numerator
                )
                for i in test_idx
            }
    except ValueError as e:
        logger.warning(f"FOLD-RM: {e}, skipping")
        return [], []
    for w in caught:
        logger.debug(f"FOLD-RM internal warn: {w.message}")

    train, test = [], []
    for i, row in enumerate(data):
        (test if i in in_test else train).append(row)
    return train, test


def _read_column(path: Path, column: str) -> list[str]:
    with open(path, newline="") as f:
        return [row[column] for row in csv.DictReader(f)]


def per_class_scores(Y_hat: list, Y: list) -> dict[str, dict]:
    """Precision/recall/support of every label."""
    result = {}
    for label in sorted(set(Y) | set(Y_hat)):
        tp = sum(1 for y, yh in zip(Y, Y_hat) if y == yh == label)
        predicted = Y_hat.count(label)
        support = Y.count(label)
        result[label] = {
            "precision": round(tp / predicted, 4) if predicted else 0.0,
            "baseline_precision": round(support / len(Y), 4),
            "recall": round(tp / support, 4) if support else 0.0,
            "support": support,
        }
    return result


def combined_scores(Y_hat: list, Y: list, default: str) -> dict:
    """Scores of the merged prediction over the original classes."""
    classes = per_class_scores(Y_hat, Y)
    present = [v for v in classes.values() if v["support"] > 0]

    labels = sorted(set(Y) | set(Y_hat), key=float)
    n = len(Y)

    accuracy = sum(y == yh for y, yh in zip(Y, Y_hat)) / n if n else 0.0

    # Macro-recall: average recall across classes with non-zero support.
    recall = sum(v["recall"] for v in present) / len(present) if present else 0.0

    # Macro-F1: average per-class F1, giving each class equal weight.
    f1_scores = [
        (
            2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"])
            if v["precision"] + v["recall"] > 0
            else 0.0
        )
        for v in present
    ]
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Weighted precision: weight each class by its ground-truth support.
    weighted_precision = (
        sum(v["precision"] * v["support"] for v in present) / n if n else 0.0
    )

    # Weighted recall: weight each class by its ground-truth support.
    weighted_recall = sum(v["recall"] * v["support"] for v in present) / n if n else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "majority_baseline": round(Y.count(default) / n, 4) if n else 0.0,
        "recall": round(recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "per_class": classes,
        # confusion[true][predicted]
        "confusion": {
            t: {p: sum(y == t and yh == p for y, yh in zip(Y, Y_hat)) for p in labels}
            for t in labels
        },
    }


def _warn_if_stale(target: str, metrics_path: Path, current: dict) -> None:
    try:
        stored = json.loads(metrics_path.read_text())
    except OSError, json.JSONDecodeError:
        logger.warning(
            f"FOLD-RM {target}: unreadable {metrics_path}, use --force learning to retrain"
        )
        return
    changed = [k for k, v in current.items() if stored.get(k) != v]
    if changed:
        logger.warning(
            f"FOLD-RM {target}: cached models differ from the current run in "
            f"{', '.join(changed)}, use --force learning to retrain"
        )


def _was_skipped(metrics_path: Path) -> bool:
    try:
        return json.loads(metrics_path.read_text()).get("combined") is None
    except OSError, json.JSONDecodeError:
        return False


def _train_target(
    Classifier,
    dataset_csv: Path,
    schema: dict[str, list[str]],
    target: str,
    target_dir: Path,
    strategy: str,
    ratio: float,
    split: str,
    test_ratio: float,
    seed: int,
    cleaner: BaseCleaner,
    gpu: bool,
) -> tuple[list[dict], dict | None]:
    loader = Classifier(
        attrs=list(schema["features"]), numeric=list(schema["numeric"]), label=target
    )
    data = loader.load_data(str(dataset_csv))
    attrs = loader.attrs  # features + label

    if split == "homogeneous":
        dates = _read_column(dataset_csv, DATE_COLUMN)
        train, test = homogeneous_split(data, dates, test_ratio, seed)
    elif split == "date":
        dates = _read_column(dataset_csv, DATE_COLUMN)
        train, test = date_split(data, dates, test_ratio, seed)
    elif split == "row":
        train, test = stratified_split(data, test_ratio, seed)
    else:
        raise ValueError(f"Unknown split {split!r}")

    if not train or not test:
        logger.warning(
            f"FOLD-RM {target}: {len(train)} train / {len(test)} test rows, too few for a split, skipping"
        )
        return [], None
    logger.info(f"FOLD-RM {target}: {len(train)} train / {len(test)} test rows")

    framing = STRATEGIES[strategy]()
    results = []
    raw_on_test = {}
    for task, task_train, task_test in framing.tasks(train, test):
        model = Classifier(attrs=attrs, numeric=list(schema["numeric"]), label=target)
        start = timer()
        framing.fit(model, task, task_train, ratio, gpu)
        # todo model.fit_no_IG_F1(task_train, ratio=ratio) per startegy
        cleaner.clean(model, task_train)
        fit_seconds = timer() - start

        Y = [d[-1] for d in task_test]
        default = framing.default_of(task, task_train)
        raw = model.predict(task_test)
        raw_on_test[task] = raw
        Y_hat = [default if y is None else y for y in raw]
        acc, p, r, f1 = scores(Y_hat, Y, weighted=True)
        classes = per_class_scores(Y_hat, Y)

        (target_dir / f"{task}.lp").write_text(model.get_asp(simple=True) + "\n")
        save_model(model, target_dir / f"{task}.pkl")

        result = {
            "task": task,
            "accuracy": round(acc, 4),
            "majority_baseline": round(Y.count(default) / len(Y), 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "per_class": classes,
            "uncovered": raw.count(None),
            "n_rules": len(model.rules),
            "fit_seconds": round(fit_seconds, 2),
        }
        recalls = " ".join(f"{k}:{v['recall']}" for k, v in classes.items())
        logger.debug(
            f"FOLD-RM {target} {task}: acc {result['accuracy']} "
            f"(baseline {result['majority_baseline']}) f1 {result['f1']} "
            f"recall {recalls} rules {result['n_rules']}"
        )
        results.append(result)

    Y = [d[-1] for d in test]
    default = majority_label(train)
    combined = combined_scores(
        framing.combine(raw_on_test, train, len(test)), Y, default
    )
    combined["n_rules"] = sum(r["n_rules"] for r in results)
    logger.info(
        f"FOLD-RM {target}: acc {combined['accuracy']} (baseline "
        f"{combined['majority_baseline']}) macro F1 {combined['macro_f1']} "
        f"rules {combined['n_rules']}"
    )
    return results, combined


def train_fold_rm(
    dataset_csv: Path,
    schema: dict[str, list[str]],
    output_dir: Path,
    *,
    strategy: str,
    ratio: float,
    split: str,
    test_ratio: float,
    seed: int,
    cleaning: dict,
    exclude_features: dict[str, list[str]],
    gpu: bool,
    force: bool,
) -> None:
    """
    Trains FOLD-RM models for every label in `schema` on the merged dataset.

    Args:
        dataset_csv: The merged dataset produced by the translator.
        schema: {"labels", "features", "categorical", "numeric"} as given by FoldRmTranslator.schema.
        output_dir: Root folder for the learned models.
        strategy: Key of STRATEGIES ("one_vs_rest", "multiclass", "ordinal"
            or "ordinal_count", "multiclass_no_ig").
        ratio: FOLD-RM exception ratio hyperparameter.
        split: "homogeneous" holds out whole days keeping the label
            proportions (best effort); "date" holds out random whole days;
            "row" is a per-label stratified split.
        test_ratio: Fraction of days (or rows) held out for testing.
        seed: Seed for the split.
        cleaning: Options of cleaning.FoldRmCleaner (min_support,
            min_exception_support, mdl_lambda, snap), applied to every model
            on its training rows.
        exclude_features: target -> feature column prefixes left out of its
            models (e.g. the cloud amounts for prev_cloud).
        gpu: Use CUDatILP's CUDA training (fitGPU).
        force: Retrain targets whose metrics already exist.
    """
    Classifier = load_classifier_cls()
    cleaner = FoldRmCleaner(**cleaning)

    current = {
        "dataset": str(dataset_csv),
        "strategy": strategy,
        "ratio": ratio,
        "split": split,
        "test_ratio": test_ratio,
        "seed": seed,
        "min_support": cleaning["min_support"],
        "min_exception_support": cleaning["min_exception_support"],
        "gpu": gpu,
    }
    # only when used, so the metrics.json of older runs stay comparable
    if cleaning["mdl_lambda"] is not None:
        current["mdl_lambda"] = cleaning["mdl_lambda"]
    if cleaning["snap"]:
        current["snap"] = True

    logger.info("Starting FOLD-RM training")
    for target in schema["labels"]:
        excluded = tuple(exclude_features.get(target, []))
        target_schema = {
            **schema,
            "features": [c for c in schema["features"] if not c.startswith(excluded)],
            "numeric": [c for c in schema["numeric"] if not c.startswith(excluded)],
        }
        target_current = {
            **current,
            **({"exclude_features": list(excluded)} if excluded else {}),
        }
        target_dir = output_dir / target
        metrics_path = target_dir / METRICS_FILE
        if not force and metrics_path.exists() and not _was_skipped(metrics_path):
            logger.info(f"FOLD-RM {target}: already trained, skipping")
            _warn_if_stale(target, metrics_path, target_current)
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        # drop models of a previous strategy
        for stale in [*target_dir.glob("*.lp"), *target_dir.glob("*.pkl")]:
            stale.unlink()
        metrics_path.unlink(missing_ok=True)

        results, combined = _train_target(
            Classifier,
            dataset_csv,
            target_schema,
            target,
            target_dir,
            strategy,
            ratio,
            split,
            test_ratio,
            seed,
            cleaner,
            gpu,
        )
        if combined is None:
            continue
        metrics = {**target_current, "combined": combined, "tasks": results}
        metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("FOLD-RM training completed.")
