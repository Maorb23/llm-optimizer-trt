import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Set up the root logger.
    """
    logger = logging.getLogger("llm_optimizer")
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_logger(name):
    """
    Get a logger instance.
    """
    return logging.getLogger(f"llm_optimizer.{name}")
