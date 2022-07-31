from pathlib import Path
from typing import Sequence

from ..C import SUFFIXES_CSV, SUFFIXES_HDF5
from .base import History, HistoryBase, HistoryTypeError
from .csv import CsvHistory
from .hdf5 import Hdf5History
from .memory import MemoryHistory


def create_history_from_options(
    self,
    id: str,
    x_names: Sequence[str],
) -> HistoryBase:
    """Create a :class:`HistoryBase` object; Factory method.

    Parameters
    ----------
    id:
        Identifier for the history.
    x_names:
        Parameter names.
    """
    # create different history types based on the inputs
    if self.storage_file is None:
        if self.trace_record:
            return MemoryHistory(options=self)
        else:
            return History(options=self)

    # replace id template in storage file
    storage_file = self.storage_file.replace("{id}", id)

    # evaluate type
    suffix = Path(storage_file).suffix[1:]

    # create history type based on storage type
    if suffix in SUFFIXES_CSV:
        return CsvHistory(x_names=x_names, file=storage_file, options=self)
    elif suffix in SUFFIXES_HDF5:
        return Hdf5History(id=id, file=storage_file, options=self)
    else:
        raise HistoryTypeError(suffix)
