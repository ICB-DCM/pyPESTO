"""
Logging
=======

Logging convenience functions.
"""

import logging


def log(
    name: str = "pypesto",
    level: int = logging.INFO,
    console: bool = True,
    filename: str = "",
):
    """
    Log messages from `name` with `level` to any combination of console/file.

    Parameters
    ----------
    name:
        The name of the logger.
    level:
        The output level to use.
    console:
        If True, messages are logged to console.
    filename:
        If specified, messages are logged to a file with this name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        logger.addHandler(ch)

    if filename:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        logger.addHandler(fh)


def log_to_console(level: int = logging.INFO):
    """
    Log to console.

    Parameters
    ----------
    See the `log` method.
    """
    log(level=level, console=True)


def log_to_file(
    level: int = logging.INFO, filename: str = ".pypesto_logging.log"
):
    """
    Log to file.

    Parameters
    ----------
    See the `log` method.
    """
    log(level=level, filename=filename)


def log_level_active(logger: logging.Logger, level: int) -> bool:
    """Check whether the requested log level is active in any handler.

    This is useful in case log expressions are costly.

    Parameters
    ----------
    logger:
        The logger.
    level:
        The requested log level.

    Returns
    --------
    active:
        Whether there is a handler registered that handles events of importance
        at least `level` and higher.
    """
    for handler in logger.handlers:
        # it is DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50 increasing
        if handler.level <= level:
            return True
        return False
