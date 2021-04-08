from ..engine import Engine, SingleCoreEngine
from ..objective import HistoryOptions
from ..store.save_to_hdf5 import get_or_create_group

import os
import h5py


def check_hdf5_mp(history_options: HistoryOptions,
                  engine: Engine):
    """
    This function checks whether we use another engine than
    `SingleCoreEngine` and whether we use a `Hdf5History`. If
    that is the case, i rephrases the filename to create a hdf5 file
    per start. Returns filename for later usage in filling hdf5 file
    otherwise return None.
    """
    if not isinstance(engine, SingleCoreEngine):
        filename = history_options.storage_file
        fn, type = os.path.splitext(history_options.storage_file)
        # creates a folder in which the hdf5 files will be saved
        if not os.path.exists(fn):
            os.mkdir(fn)
        fn = fn + '/' + fn + '_{id}'
        history_options.storage_file = fn + type
        # create hdf5 file that gather the other with history group
        f = h5py.File(filename, mode='a')
        get_or_create_group(f, "history")
        return filename
    return None


def fill_hdf5_file(ret,
                   filename):
    f = h5py.File(filename, mode='a')
    for result in ret:
        id = result['id']
        if f'history/{id}' in f:
            del f[f'history/{id}']
        f[f'history/{id}'] = h5py.ExternalLink(result['history'].file,
                                               f'history/{id}')
