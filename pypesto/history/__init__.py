"""
History
=======

Objetive function call history. The history tracks and stores function
evaluations performed by e.g. the optimizer and other routines, allowing to
e.g. recover results from failed runs, fill in further details,
and evaluate performance.
"""

from .base import History, HistoryBase
from .csv import CsvHistory
from .hdf5 import Hdf5History
from .memory import MemoryHistory
from .optimizer import OptimizerHistory
from .options import HistoryOptions
