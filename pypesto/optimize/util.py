"""Utility functions for :py:func:`pypesto.optimize.minimize`."""
import datetime
import os
import binascii
from pathlib import Path
from typing import List
import h5py

from ..engine import Engine, SingleCoreEngine
from ..objective import HistoryOptions
from ..store.save_to_hdf5 import get_or_create_group
from ..store import write_result
from ..result import Result
from .optimizer import OptimizerResult


def preprocess_hdf5_history(
    history_options: HistoryOptions,
    engine: Engine,
):
    """Create a folder for partial HDF5 files if parallelization is used.

    This is because single hdf5 file access is not thread-safe.

    Parameters
    ----------
    engine:
        The Engine which is used in the optimization.
    history_options:
        The HistoryOptions used in the optimization.

    Returns
    -------
    history_requires_postprocessing:
        Whether history storage post-processing is required.
    """
    storage_file = history_options.storage_file

    # nothing to do if no history stored
    if storage_file is None:
        return False

    # extract storage type
    path = Path(storage_file)

    # nothing to do if csv history and correctly set
    if path.suffix == ".csv":
        if "{id}" not in storage_file:
            raise ValueError(
                "For csv history, the `storage_file` must contain an `{id}` "
                "template"
            )
        return False

    # assuming hdf5 history henceforth
    if path.suffix not in [".h5", ".hdf5"]:
        raise ValueError(
            "Only history storage to '.csv' and '.hdf5' is supported, got "
            f"{path.suffix}",
        )

    # nothing to do if no parallelization
    if isinstance(engine, SingleCoreEngine):
        return False

    # create directory with same name as original file stem
    template_path = (
        path.parent / path.stem / (path.stem + "_{id}" + path.suffix)
    )
    template_path.parent.mkdir(parents=True, exist_ok=True)
    # set history file to template path
    history_options.storage_file = str(template_path)

    return True


def postprocess_hdf5_history(
    ret: List[OptimizerResult],
    storage_file: str,
    history_options: HistoryOptions,
) -> None:
    """Create single history file pointing to files of multiple starts.

    Create links in `storage_file` to the history of each start contained in
    `ret`, the results of the optimization.

    Parameters
    ----------
    ret:
        The result iterable returned by the optimization.
    storage_file:
        The filename of the hdf5 file in which the histories
        are to be gathered.
    history_options:
        History options used in the optimization.
    """
    # create hdf5 file that gathers the others within history group
    with h5py.File(storage_file, mode='w') as f:
        # create file and group
        get_or_create_group(f, "history")
        # append links to each single result file
        for result in ret:
            id = result['id']
            f[f'history/{id}'] = h5py.ExternalLink(
                result['history'].file,
                f'history/{id}'
            )

    # reset storage file (undo preprocessing changes)
    history_options.storage_file = storage_file


def autosave(filename: str,
             result: Result,
             store_type: str,
             overwrite: bool = False):
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
    store_type:
        Either `optimize`, `sample` or `profile`. Depending on the
        method the function is called in.
    overwrite:
        Whether to overwrite the currently existing results.
    """
    if filename is None:
        return

    if filename == "Auto":
        time = datetime.datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
        filename = time+f"_{store_type}_result_" \
                        f"{binascii.b2a_hex(os.urandom(8)).decode()}.h5"
    # set the type to True and pass it on to write_result
    to_save = {store_type: True}
    write_result(result=result,
                 overwrite=overwrite,
                 filename=filename,
                 **to_save)
