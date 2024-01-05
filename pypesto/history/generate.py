"""Generate a history from options and inputs."""

from pathlib import Path
from typing import Sequence

from ..C import SUFFIXES_CSV, SUFFIXES_HDF5
from .amici import CsvAmiciHistory, Hdf5AmiciHistory
from .base import CountHistory, HistoryBase
from .csv import CsvHistory
from .hdf5 import Hdf5History
from .memory import MemoryHistory
from .options import HistoryOptions
from .util import HistoryTypeError


def create_history(
    id: str,
    x_names: Sequence[str],
    options: HistoryOptions,
    amici_objective: bool,
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
    amici_objective:
        Indicates if AmiciObjective was used

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
        if amici_objective:
            return CsvAmiciHistory(
                x_names=x_names, file=storage_file, options=options
            )
        else:
            return CsvHistory(
                x_names=x_names, file=storage_file, options=options
            )
    elif suffix in SUFFIXES_HDF5:
        if amici_objective:
            return Hdf5AmiciHistory(id=id, file=storage_file, options=options)
        else:
            return Hdf5History(id=id, file=storage_file, options=options)
    else:
        raise HistoryTypeError(suffix)
