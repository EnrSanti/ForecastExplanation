import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

import data_extraction
import features_detection
import ground_truth
import learning
import reasoning
import translators
from config import CUT, RunConfig, Stage, load_runs

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("ForecastExplanation")


def parse_args() -> argparse.Namespace:
    stages = [s.value for s in Stage]
    parser = argparse.ArgumentParser(description="ForecastExplanation Pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--force",
        choices=stages,
        help="Recompute this stage and every stage after it, ignoring cached results",
    )
    parser.add_argument(
        "--stop-after",
        choices=[CUT, *stages],
        help="Last stage to run ('cut' = only download and cut the GRIB files)",
    )
    parser.add_argument(
        "--clean",
        nargs="+",
        choices=["grib", "cut", "extracted"],
        help="Artefacts to delete after the run",
    )
    parser.add_argument(
        "--clustering",
        action=argparse.BooleanOptionalAction,
        help="Run TOBAC on clustered data instead of the discrete one",
    )
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        help="Generate visualization images of the tracking results",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action=argparse.BooleanOptionalAction,
        help="Enable debug logging for the application",
    )
    parser.add_argument("--workers", type=int, help="Parallel worker processes")
    return parser.parse_args()


def run_pipeline(cfg: RunConfig) -> None:
    region = cfg.build_region()
    output_path = cfg.output_path

    data_extraction.extract(
        cfg.dates,
        region,
        output_path=output_path,
        clustering=cfg.clustering,
        force_redo=cfg.forces(Stage.DATA),
        just_cut=cfg.stop_after == CUT,
        create_images=cfg.save_images,
        workers=cfg.workers,
    )
    if not cfg.runs(Stage.FEATURES):
        return

    input_dir = output_path / (
        data_extraction.CLUSTERED_DATA_DIR
        if cfg.clustering
        else data_extraction.DISCRETE_DATA_DIR
    )
    features_detection.run_tobac(
        cfg.dates,
        input_dir=input_dir,
        output_dir=output_path,
        region=region,
        force=cfg.forces(Stage.FEATURES),
        save_images=cfg.save_images,
        workers=cfg.workers,
    )
    if not cfg.runs(Stage.REASONING):
        return

    reasoning.reason(
        cfg.dates,
        output_path,
        output_path,
        region,
        force=cfg.forces(Stage.REASONING),
        workers=cfg.workers,
    )
    ground_truth.generate_gt(cfg.dates, output_path, force=cfg.forces(Stage.REASONING))
    if not cfg.runs(Stage.TRANSLATION):
        return

    dataset_csv = translators.FoldRmTranslator().translate(
        cfg.dates,
        output_path,
        output_path / "translated",
        region,
        force=cfg.forces(Stage.TRANSLATION),
        workers=cfg.workers,
    )
    if not cfg.runs(Stage.LEARNING) or not dataset_csv:
        return

    learning.train_fold_rm(
        dataset_csv,
        translators.FoldRmTranslator.schema_from_csv(dataset_csv),
        output_path / "fold_rm",
        **cfg.fold_rm.model_dump(),
        force=cfg.forces(Stage.LEARNING),
        verbose=cfg.debug,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    cli_overrides = {
        k: v for k, v in vars(args).items() if k != "config" and v is not None
    }

    try:
        runs = load_runs(args.config, cli_overrides)
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    for run_name, cfg in runs.items():
        level = logging.DEBUG if cfg.debug else logging.INFO
        for name in ("ForecastExplanation", "data_extraction", "features_detection"):
            logging.getLogger(name).setLevel(level)

        logger.info(f" --- Starting {run_name} ---")
        try:
            run_pipeline(cfg)
        except Exception:
            logger.exception(f"Error running {run_name}")
            continue

        if cfg.clean:
            data_extraction.clean_artifacts(cfg.dates, cfg.output_path, cfg.clean)
        logger.info(f"--- Finished {run_name} ---\n\n")


if __name__ == "__main__":
    main()
