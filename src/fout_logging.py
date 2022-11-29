import logging
import sys
import os
from contextlib import contextmanager
from typing import List

from tqdm import tqdm

log_folder = "logs"
loggers = dict()
silenced_loggers = list()
non_silenced_loggers = list()

OKAY_LEVEL_NUM = 9
SUCCESS_LEVEL_NUM = 13
END_LEVEL_NUM = 11
START_LEVEL_NUM = 12
logging.addLevelName(OKAY_LEVEL_NUM, "OKAY")
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
logging.addLevelName(END_LEVEL_NUM, "END")
logging.addLevelName(START_LEVEL_NUM, "START")


def okay(self, message, *args, **kwargs):
    self._log(OKAY_LEVEL_NUM, message, args, **kwargs)


def success(self, message, *args, **kwargs):
    self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)


def end(self, message, *args, **kwargs):
    self._log(END_LEVEL_NUM, message, args, **kwargs)


def start(self, message, *args, **kwargs):
    self._log(START_LEVEL_NUM, message, args, **kwargs)


logging.Logger.success = success
logging.Logger.okay = okay
logging.Logger.start = start
logging.Logger.end = end


class ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\033[38m"
    yellow = "\033[33m"
    orange = "\033[40m"
    red = "\033[31m"
    bold_red = "\033[31;1m"
    reset = "\033[0m"
    blue = '\033[94m'
    cyan = '\033[96m'
    green = '\033[92m'
    marine_blue = '\033[36m'
    dark_gray = '\033[38;5;231m'


    FORMATS = {
        logging.DEBUG: yellow,
        logging.INFO: grey,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
        OKAY_LEVEL_NUM: cyan,
        SUCCESS_LEVEL_NUM: green,
        START_LEVEL_NUM: dark_gray,
        END_LEVEL_NUM: marine_blue,
    }

    def format(self, record):
        color = self.FORMATS.get(record.levelno)
        fmt = color + self._fmt + self.reset
        formatter = logging.Formatter(fmt, datefmt=self.datefmt)
        return formatter.format(record)


def setup_custom_logger(name):
    formatter = ColoredFormatter(fmt='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
                                 datefmt='[%Y-%m-%d %H:%M:%S]')
    filename = name.lower().replace(" ", "_")
    log_file = os.path.join(log_folder, "logs" + '.log')

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handler = logging.FileHandler(log_file, mode='w+')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


class SilentFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.levelno in {logging.DEBUG,
                                      logging.INFO,
                                      logging.WARNING,
                                      OKAY_LEVEL_NUM,
                                      SUCCESS_LEVEL_NUM,
                                      START_LEVEL_NUM,
                                      END_LEVEL_NUM}


class Logger:
    def __init__(self, name):
        self._logger = setup_custom_logger(name)

    def silence(self):
        self._logger.addFilter(SilentFilter())

    def error(self, message):
        self.e(message)
        raise RuntimeError(message)

    def e(self, message):
        self._logger.error(message)

    def w(self, message):
        self._logger.warning(message)

    def i(self, message):
        self._logger.info(message)

    def s(self, message):
        self._logger.success(message)

    def ok(self, message):
        self._logger.okay(message)

    def debug(self, message):
        self._logger.debug(message)

    def start(self, message):
        self._logger.start(message)

    def end(self, message=None):
        if message is not None:
            self._logger.end(message)
        self._logger.end("--------------------------------------")


class LoggerMixin:
    def __init__(self, *args, **kwargs):
        super(LoggerMixin, self).__init__(*args, **kwargs)
        class_name = self.__class__.__name__
        if class_name not in loggers:
            self.log = Logger(class_name)
            loggers[class_name] = self.log
            if class_name in silenced_loggers:
                self.log.silence()
            if len(non_silenced_loggers) > 0 and class_name not in non_silenced_loggers:
                self.log.silence()
        else:
            self.log = loggers[class_name]


def silence(classes: List[type]):
    silenced_loggers.extend([cls.__name__ for cls in classes])


def silence_all_but(classes: List[type]):
    non_silenced_loggers.extend([cls.__name__ for cls in classes])


tqdm_disabled=False


def my_tqdm(arg):
    return tqdm(arg, file=sys.stdout, colour='green', disable=tqdm_disabled)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
