"""Utility functions for :py:func:`pypesto.optimize.minimize`."""
import datetime

from ..engine import Engine, SingleCoreEngine
from ..objective import HistoryOptions
from ..store.save_to_hdf5 import get_or_create_group
from ..store import write_result
from ..result import Result
from pathlib import Path
from typing import Union


import h5py


def check_hdf5_mp(
    history_options: HistoryOptions,
    engine: Engine,
) -> Union[str, None]:
    """
    Create a folder for partial HDF5 files.

    If no parallelization engine is used, do nothing.

    Parameters
    ----------
    engine:
        The Engine which is used in the optimization
    history_options:
        The HistoryOptions used in the optimization

    Returns
    -------
    The filename that will be used to combine the partial HDF5 files later.
    If a parallelization engine is not used, `None` is returned.
    """
    if isinstance(engine, SingleCoreEngine):
        return None
    filename = history_options.storage_file
    file_path = Path(filename)

    # create directory with same name as original file stem
    partial_file_path = (
            file_path.parent / file_path.stem /
            (file_path.stem + '_{id}' + file_path.suffix)
    )
    partial_file_path.parent.mkdir(parents=True, exist_ok=True)
    history_options.storage_file = str(partial_file_path)

    # create hdf5 file that gathers the others within history group
    with h5py.File(filename, mode='a') as f:
        get_or_create_group(f, "history")
    return filename


def fill_hdf5_file(
    ret: list,
    filename: str
) -> None:
    """
    Create single history file pointing to files of multiple starts.

    Create links in `filename` to the
    history of each start contained in ret, the results
    of the optimization.

    Parameters
    ----------
    ret:
        The result iterable returned by the optimization.
    filename:
        The filename of the hdf5 file in which the histories
        are to be gathered.
    """
    with h5py.File(filename, mode='a') as f:
        for result in ret:
            id = result['id']
            f[f'history/{id}'] = h5py.ExternalLink(
                result['history'].file,
                f'history/{id}'
            )


def autosave(filename: str, result: Result,
             store_type: str, overwrite: bool = False):
    """
    Save the result of optimization, profiling or sampling automatically.

    Parameters
    ----------
    filename:
        Either the filename to save to or "Auto", in which case it will
        automatically generate a file named
        `year_month_day_{type}_result.hdf5`.
    result:
        The result to be saved.
    type:
        Either `optimization`, `sampling` or `profiling`. Depending on the
        method the function is called in.
    overwrite:
        Whether to overwrite the currently existing results.
    """
    if filename is None:
        return None
    if filename == "Auto":
        time = datetime.datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
        filename = time+f"_{store_type}_result.h5"
    # set the type to True and pass it on to write_result
    to_save = {store_type: True}
    write_result(result=result,
                 overwrite=overwrite,
                 filename=filename,
                 **to_save)
