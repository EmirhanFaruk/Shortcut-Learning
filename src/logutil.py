"""
logutil.py
----------
One place to configure timestamped logging. Everything (config, epochs with
durations, checkpoint saves/deletes, eval results, total runtime) goes through a
logger so the run is fully auditable from the log file.
"""
import logging
import os
import time
from datetime import datetime


def setup_logger(log_dir, name="experiment"):
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{stamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()           # idempotent if called twice
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"logging to {log_path}")
    return logger


class Timer:
    """Context manager that logs how long a labelled block took."""
    def __init__(self, logger, label):
        self.logger, self.label = logger, label

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info(f"START  {self.label}")
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        self.logger.info(f"DONE   {self.label}  ({dt:.1f}s)")


def fmt_secs(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"
