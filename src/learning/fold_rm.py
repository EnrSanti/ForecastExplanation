import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from timeit import default_timer as timer

from .cudatilp import fit_target, load_classifier_cls, prune_rules, save_model, scores
from .strategies import STRATEGIES, majority_label

logger = logging.getLogger("ForecastExplanation")

METRICS_FILE = "metrics.json"
SPLITS = ("date", "row")


def stratified_split(
    data: list[list], test_ratio: float, seed: int
) -> tuple[list[list], list[list]]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

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
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    days = sorted(set(dates))
    random.Random(seed).shuffle(days)
    test_days = set(days[: round(len(days) * test_ratio)])

    train, test = [], []
    for row, day in zip(data, dates):
        (test if day in test_days else train).append(row)
    return train, test


def _read_column(path: Path, column: str) -> list[str]:
    with open(path, newline="") as f:
        return [row[column] for row in csv.DictReader(f)]


def per_class_scores(Y_hat: list, Y: list) -> dict[str, dict]:
    """Precision/recall/support of every label, so a model that never predicts
    its minority class can't hide behind the majority class's accuracy.
    `baseline_precision` is the label's share of the rows: the precision of
    guessing it without looking at the features."""
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
    """Scores of the merged prediction over the original classes. Macro F1 and
    balanced accuracy average over the classes present in the test set, so
    every class weighs the same however rare it is."""
    classes = per_class_scores(Y_hat, Y)
    present = [v for k, v in classes.items() if v["support"]]
    f1 = [
        (
            2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"])
            if v["precision"] + v["recall"]
            else 0.0
        )
        for v in present
    ]
    labels = sorted(set(Y) | set(Y_hat), key=float)
    return {
        "accuracy": round(sum(y == yh for y, yh in zip(Y, Y_hat)) / len(Y), 4),
        "majority_baseline": round(Y.count(default) / len(Y), 4),
        "macro_f1": round(sum(f1) / len(f1), 4),
        "balanced_accuracy": round(sum(v["recall"] for v in present) / len(present), 4),
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
    min_support: int,
    min_exception_support: int,
    gpu: bool,
    verbose: bool,
) -> tuple[list[dict], dict | None]:
    # load_data reads columns in CSV order but names them after `attrs`, so
    # the features must be passed in header order
    loader = Classifier(
        attrs=list(schema["features"]), numeric=list(schema["numeric"]), label=target
    )
    data = loader.load_data(str(dataset_csv))
    attrs = loader.attrs  # features + label

    if split == "date" and schema.get("date"):
        dates = _read_column(dataset_csv, schema["date"])
        if len(dates) != len(data):
            raise ValueError(f"{dataset_csv}: {len(dates)} dates for {len(data)} rows")
        train, test = date_split(data, dates, test_ratio, seed)
    else:
        if split == "date":
            logger.warning(
                f"FOLD-RM {target}: no date column in {dataset_csv.name}, "
                "falling back to a stratified row split"
            )
        train, test = stratified_split(data, test_ratio, seed)
    logger.info(f"FOLD-RM {target}: {len(train)} train / {len(test)} test rows")
    if not test:
        logger.warning(f"FOLD-RM {target}: too few rows for a test split, skipping")
        return [], None

    framing = STRATEGIES[strategy]()
    results = []
    raw_on_test = {}
    for task, task_train, task_test in framing.tasks(train, test):
        model = Classifier(attrs=attrs, numeric=list(schema["numeric"]), label=target)
        positive = framing.target_of(task)
        start = timer()
        if positive:
            if gpu:
                logger.warning(f"FOLD-RM {target} {task}: trained on CPU, ignoring gpu")
            fit_target(model, task_train, positive, ratio)
        elif gpu:
            model.fitGPU(task_train, ratio=ratio, verbose=verbose)
        else:
            model.fit(task_train, ratio=ratio, verbose=verbose)
        prune_rules(model, task_train, min_support, min_exception_support)
        fit_seconds = timer() - start

        Y = [d[-1] for d in task_test]
        # uncovered examples fall back to the majority label of the training
        # set, or to the other side of the task when rules target one label
        if positive:
            other = {d[-1] for d in task_train} - {positive}
            default = other.pop() if other else positive
        else:
            default = majority_label(task_train)
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
            "majority_baseline": round(Y.count(default) / len(Y), 4) if Y else None,
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
        f"balanced acc {combined['balanced_accuracy']} rules {combined['n_rules']}"
    )
    return results, combined


def train_fold_rm(
    dataset_csv: str | Path,
    schema: dict[str, list[str]],
    output_dir: str | Path,
    *,
    strategy: str = "one_vs_rest",
    ratio: float = 0.7,
    split: str = "date",
    test_ratio: float = 0.3,
    seed: int = 42,
    min_support: int = 0,
    min_exception_support: int = 0,
    gpu: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """
    Trains FOLD-RM models for every label in `schema` on the merged dataset.

    Each target excludes all label columns from its features, so one label
    never leaks into another. Rules (.lp), pickled models and metrics.json are
    written to `output_dir/<target>/`.

    Args:
        dataset_csv: The merged dataset produced by the translator.
        schema: {"labels", "features", "categorical", "numeric"} as given by FoldRmTranslator.schema.
        output_dir: Root folder for the learned models.
        strategy: Key of STRATEGIES ("one_vs_rest", "multiclass" or "ordinal").
        ratio: FOLD-RM exception ratio hyperparameter.
        split: "date" holds out whole days (falls back to "row" when the
            dataset has no date column); "row" is a per-label stratified split.
        test_ratio: Fraction of days (or rows) held out for testing.
        seed: Seed for the split.
        min_support: Drop the rules holding for fewer training rows of their
            own label (0 keeps every rule).
        min_exception_support: Drop the exceptions correcting fewer training
            rows (0 keeps every exception).
        gpu: Use CUDatILP's CUDA training (fitGPU).
        force: Retrain targets whose metrics already exist.
        verbose: Print CUDatILP's per-phase timing breakdown to stdout.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}, expected one of {list(SPLITS)}")
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}, expected one of {list(STRATEGIES)}"
        )

    dataset_csv = Path(dataset_csv)
    output_dir = Path(output_dir)
    Classifier = load_classifier_cls()

    current = {
        "dataset": str(dataset_csv),
        "strategy": strategy,
        "ratio": ratio,
        "split": split,
        "test_ratio": test_ratio,
        "seed": seed,
        "min_support": min_support,
        "min_exception_support": min_exception_support,
        "gpu": gpu,
    }

    logger.info("Starting FOLD-RM training")
    for target in schema["labels"]:
        target_dir = output_dir / target
        metrics_path = target_dir / METRICS_FILE
        if not force and metrics_path.exists():
            logger.info(f"FOLD-RM {target}: already trained, skipping")
            _warn_if_stale(target, metrics_path, current)
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        # models of a previous strategy would sit next to the new ones
        for stale in [*target_dir.glob("*.lp"), *target_dir.glob("*.pkl")]:
            stale.unlink()

        results, combined = _train_target(
            Classifier,
            dataset_csv,
            schema,
            target,
            target_dir,
            strategy,
            ratio,
            split,
            test_ratio,
            seed,
            min_support,
            min_exception_support,
            gpu,
            verbose,
        )
        metrics = {**current, "combined": combined, "tasks": results}
        metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("FOLD-RM training completed.")
