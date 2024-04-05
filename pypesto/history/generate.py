"""Generate a history from options and inputs."""

from collections.abc import Sequence
from pathlib import Path

from ..C import SUFFIXES_CSV, SUFFIXES_HDF5
from .base import CountHistory, HistoryBase
from .csv import CsvHistory
from .hdf5 import Hdf5History
from .memory import MemoryHistory
from .options import HistoryOptions
from .util import HistoryTypeError


def create_history(
    id: str, x_names: Sequence[str], options: HistoryOptions
) -> HistoryBase:
    """Create a :class:`HistoryBase` object; Factory method.

    Parameters
    ----------
    id:
        Identifier for the history.
    x_names:
        Parameter names.
    options:
        History options.

    Returns
    -------
    A history object corresponding to the inputs.
    """
    # create different history types based on the inputs
    if options.storage_file is None:
        if options.trace_record:
            return MemoryHistory(options=options)
        else:
            return CountHistory(options=options)

    # replace id template in storage file
    storage_file = options.storage_file.replace("{id}", id)

    # evaluate type
    suffix = Path(storage_file).suffix[1:]

    # create history type based on storage type
    if suffix in SUFFIXES_CSV:
        return CsvHistory(x_names=x_names, file=storage_file, options=options)
    elif suffix in SUFFIXES_HDF5:
        return Hdf5History(id=id, file=storage_file, options=options)
    else:
        raise HistoryTypeError(suffix)
