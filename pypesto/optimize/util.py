from ..engine import Engine, SingleCoreEngine
from ..objective import HistoryOptions
from ..store.save_to_hdf5 import get_or_create_group
from pathlib import Path


import os
import h5py


def check_hdf5_mp(history_options: HistoryOptions,
                  engine: Engine):
    """
    This function checks whether we use another engine than
    `SingleCoreEngine` and whether we use a `Hdf5History`. If
    that is the case, it rephrases the filename to create an hdf5 file
    per start. Returns filename for later usage in filling hdf5 file
    otherwise return None.

    Paramters
    ---------
    engine:
        The Engine which is used in the optimization
    history_options:
        The HistoryOptions used in the optimization

    Returns
    -------
    filename:
        string containing the original filename.
    """
    if not isinstance(engine, SingleCoreEngine):
        filename = history_options.storage_file
        file_path = Path(filename)
        fn = file_path.parent / (file_path.stem)
        # create directory with same name as original file
        fn.mkdir(parents=True, exist_ok=True)
        fn = fn / (file_path.stem + '_{id}' + file_path.suffix)
        fn = str(fn)
        history_options.storage_file = fn + type
        # create hdf5 file that gathers the others within history group
        with h5py.File(filename, mode='a') as f:
            get_or_create_group(f, "history")
        return filename
    return None


def fill_hdf5_file(ret,
                   filename):
    """
    This function creates links in `filename` to the
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
            if f'history/{id}' in f:
                del f[f'history/{id}']
            f[f'history/{id}'] = h5py.ExternalLink(
                result['history'].file,
                f'history/{id}'
            )
