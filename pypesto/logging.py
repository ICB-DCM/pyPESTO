"""
Logging
=======

Logging convenience functions.
"""

import logging


def log_to_console(level: int = None, child: str = None):
    """
    Log to console.

    Parameters
    ----------

    level:
        The output level to use. Default: logging.DEBUG.

    child:
        The name of the descendant to the 'pypesto' logger. For example, if
        child is 'model_selection', then the logger name will be
        'pypesto.model_selection'.

    """
    if level is None:
        level = logging.DEBUG

    logger = logging.getLogger('pypesto')
    if child is not None:
        logger = logger.getChild(child)

    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)


def log_to_file(level: int = None, filename: str = None, child: str = None):
    """
    Log to file.

    Parameters
    ----------

    level:
        The output level to use. Default: logging.DEBUG.

    filename:
        The name of the file to append to.
        Default: .pypesto_logging.log.

    child:
        The name of the descendant to the 'pypesto' logger. For example, if
        child is 'model_selection', then the logger name will be
        'pypesto.model_selection'.
    """

    if level is None:
        level = logging.DEBUG

    if filename is None:
        filename = ".pypesto_logging.log"

    logger = logging.getLogger('pypesto')
    if child is not None:
        logger = logger.getChild(child)

    logger.setLevel(level)
    fh = logging.FileHandler(filename)
    fh.setLevel(level)
    logger.addHandler(fh)
