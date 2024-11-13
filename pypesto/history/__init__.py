"""
History
=======

Objetive function call history. The history tracks and stores function
evaluations performed by e.g. the optimizer and other routines, allowing to
e.g. recover results from failed runs, fill in further details,
and evaluate performance.
"""

from .amici import CsvAmiciHistory, Hdf5AmiciHistory
from .base import CountHistory, CountHistoryBase, HistoryBase, NoHistory
from .csv import CsvHistory
from .generate import create_history
from .hdf5 import Hdf5History
from .memory import MemoryHistory
from .optimizer import OptimizerHistory
from .options import HistoryOptions
from .util import CsvHistoryTemplateError, HistoryTypeError
