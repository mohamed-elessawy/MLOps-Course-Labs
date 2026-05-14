import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("churn-api")

    return logger
