import logging


def add_filehandler_logger(logger, log_dir):

    filehandler = logging.FileHandler(log_dir, "w")
    formatter = logging.Formatter("%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s", datefmt="%I:%M:%S")
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)  # set the new handler

    return logger
