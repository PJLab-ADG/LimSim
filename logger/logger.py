import logging
import sys

APP_LOGGER_NAME: str = 'APP'


def setup_app_level_logger(logger_name: str = APP_LOGGER_NAME,
                           level: str = 'DEBUG',
                           use_stdout: bool = False,
                           file_name: str = "app_debug.log") -> logging.Logger:
    """create a logger

    Args:
        logger_name (str, optional): name of the logger. Defaults to APP_LOGGER_NAME.
        level (str, optional): controls the output level. Defaults to 'DEBUG'.
        use_stdout (str, optional): Whether output log to stdout. Defaults to False.
        file_name (str, optional): path where the log is saved. Defaults to "app_debug.log".

        level option: {
            'CRITICAL': CRITICAL,
            'FATAL': FATAL,
            'ERROR': ERROR,
            'WARN': WARNING,
            'WARNING': WARNING,
            'INFO': INFO,
            'DEBUG': DEBUG,
            'NOTSET': NOTSET,}
    Returns:
        logging.Logger: the logger object
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(levelname)-s]:%(filename)s %(funcName)s [Line %(lineno)s] - %(message)s")

    # output log to file
    file_handler = logging.FileHandler(file_name, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # output to stdout
    if use_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """obtain the module's logger name

    Args:
        module_name (str): name of the logger.

    Returns:
        logging.Logger: the logger object

    """
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)
