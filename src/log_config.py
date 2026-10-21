import logging
from concurrent.futures import ProcessPoolExecutor

LOG_FORMAT = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
PROJECT_LOGGERS = ("ForecastExplanation", "data_extraction", "features_detection")


def setup_logging(level: int) -> None:
    """Third-party loggers stay at ERROR, the project ones log at `level`."""
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
    for name in PROJECT_LOGGERS:
        logging.getLogger(name).setLevel(level)


def worker_pool(workers: int) -> ProcessPoolExecutor:
    """
    ProcessPoolExecutor whose workers log like the main process: they are
    started by forkserver, so they don't inherit the levels set in main().
    """
    level = logging.getLogger(PROJECT_LOGGERS[0]).getEffectiveLevel()
    return ProcessPoolExecutor(
        max_workers=workers, initializer=setup_logging, initargs=(level,)
    )
