import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from timeit import default_timer as timer

from .cudatilp import load_classifier_cls, save_model, scores
from .strategies import STRATEGIES, majority_label

logger = logging.getLogger("ForecastExplanation")

METRICS_FILE = "metrics.json"


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


def _train_target(
    Classifier,
    dataset_csv: Path,
    schema: dict[str, list[str]],
    target: str,
    target_dir: Path,
    strategy: str,
    ratio: float,
    test_ratio: float,
    seed: int,
    gpu: bool,
    verbose: bool,
) -> list[dict]:
    # load_data reads columns in CSV order but names them after `attrs`, so
    # the features must be passed in header order
    loader = Classifier(
        attrs=list(schema["features"]), numeric=list(schema["numeric"]), label=target
    )
    data = loader.load_data(str(dataset_csv))
    attrs = loader.attrs  # features + label

    train, test = stratified_split(data, test_ratio, seed)
    logger.info(f"FOLD-RM {target}: {len(train)} train / {len(test)} test rows")
    if not test:
        logger.warning(f"FOLD-RM {target}: too few rows for a test split, skipping")
        return []

    results = []
    for task, task_train, task_test in STRATEGIES[strategy]().tasks(train, test):
        model = Classifier(attrs=attrs, numeric=list(schema["numeric"]), label=target)
        start = timer()
        if gpu:
            model.fitGPU(task_train, ratio=ratio, verbose=verbose)
        else:
            model.fit(task_train, ratio=ratio, verbose=verbose)
        fit_seconds = timer() - start

        Y = [d[-1] for d in task_test]
        # uncovered examples fall back to the majority label of the training set
        default = majority_label(task_train)
        Y_hat = [default if y is None else y for y in model.predict(task_test)]
        acc, p, r, f1 = scores(Y_hat, Y, weighted=True)

        (target_dir / f"{task}.lp").write_text(model.get_asp(simple=True) + "\n")
        save_model(model, target_dir / f"{task}.pkl")

        result = {
            "task": task,
            "accuracy": round(acc, 4),
            "majority_baseline": round(Y.count(default) / len(Y), 4) if Y else None,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "n_rules": len(model.rules),
            "fit_seconds": round(fit_seconds, 2),
        }
        logger.debug(
            f"FOLD-RM {target} {task}: acc {result['accuracy']} "
            f"(baseline {result['majority_baseline']}) f1 {result['f1']} "
            f"rules {result['n_rules']}"
        )
        results.append(result)
    return results


def train_fold_rm(
    dataset_csv: str | Path,
    schema: dict[str, list[str]],
    output_dir: str | Path,
    *,
    strategy: str = "one_vs_rest",
    ratio: float = 0.7,
    test_ratio: float = 0.3,
    seed: int = 42,
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
        strategy: Key of STRATEGIES ("one_vs_rest" or "multiclass").
        ratio: FOLD-RM exception ratio hyperparameter.
        test_ratio: Fraction of rows held out for testing.
        seed: Seed for the stratified split.
        gpu: Use CUDatILP's CUDA training (fitGPU).
        force: Retrain targets whose metrics already exist.
        verbose: Print CUDatILP's per-phase timing breakdown to stdout.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}, expected one of {list(STRATEGIES)}"
        )

    dataset_csv = Path(dataset_csv)
    output_dir = Path(output_dir)
    Classifier = load_classifier_cls()

    logger.info("Starting FOLD-RM training")
    for target in schema["labels"]:
        target_dir = output_dir / target
        metrics_path = target_dir / METRICS_FILE
        if not force and metrics_path.exists():
            logger.info(f"FOLD-RM {target}: already trained, skipping")
            continue
        target_dir.mkdir(parents=True, exist_ok=True)

        results = _train_target(
            Classifier,
            dataset_csv,
            schema,
            target,
            target_dir,
            strategy,
            ratio,
            test_ratio,
            seed,
            gpu,
            verbose,
        )
        metrics = {
            "dataset": str(dataset_csv),
            "strategy": strategy,
            "ratio": ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "gpu": gpu,
            "tasks": results,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("FOLD-RM training completed.")
