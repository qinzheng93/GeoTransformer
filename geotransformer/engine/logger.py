import logging

import coloredlogs


def create_logger(log_file=None):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level=logging.DEBUG)
    logger.propagate = False

    format_str = '[%(asctime)s] [%(levelname).4s] %(message)s'

    stream_handler = logging.StreamHandler()
    colored_formatter = coloredlogs.ColoredFormatter(format_str)
    stream_handler.setFormatter(colored_formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class Logger:
    def __init__(self, log_file=None, local_rank=-1):
        if local_rank == 0 or local_rank == -1:
            self.logger = create_logger(log_file=log_file)
        else:
            self.logger = None

    def debug(self, message):
        if self.logger is not None:
            self.logger.debug(message)

    def info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def warning(self, message):
        if self.logger is not None:
            self.logger.warning(message)

    def error(self, message):
        if self.logger is not None:
            self.logger.error(message)

    def critical(self, message):
        if self.logger is not None:
            self.logger.critical(message)
