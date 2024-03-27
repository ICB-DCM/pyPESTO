"""Auto-saving."""

import binascii
import datetime
import logging
import os
from pathlib import Path
from typing import Callable, Union

import h5py

from ..result import Result
from .save_to_hdf5 import write_result

logger = logging.getLogger(__name__)


def autosave(
    filename: Union[Path, str, Callable, None],
    result: Result,
    store_type: str,
    overwrite: bool = False,
):
    """
    Save the result of optimization, profiling or sampling automatically.

    Parameters
    ----------
    filename:
        Either the filename to save to or "Auto", in which case it will
        automatically generate a file named
        `year_month_day_{type}_result.hdf5`.
        A method can also be provided. All input to the autosave method will
        be passed to the filename method. The output should be the filename
        (`str`).
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

    if isinstance(filename, Path):
        filename = str(filename)

    if filename == "Auto":
        filename = default_filename
    elif isinstance(filename, str):
        if os.path.exists(filename) and not overwrite:
            with h5py.File(filename, "r") as f:
                storage_used = store_type in f.keys()
            if storage_used:
                logger.warning(
                    f"There is already a {store_type}-result saved in "
                    f"{filename}. Please choose a different filename or set "
                    f"overwrite=True. File will be saved as in AUTO mode."
                )
                filename = default_filename
    if not isinstance(filename, str):
        filename = filename(
            result=result,
            store_type=store_type,
            overwrite=overwrite,
        )
    # set the type to True and pass it on to write_result
    to_save = {store_type: True}
    write_result(
        result=result, overwrite=overwrite, filename=filename, **to_save
    )


def default_filename(**kwargs) -> str:
    """Create a filename when results will be autosaved.

    See :func:`autosave` for additional information.
    """
    time = datetime.datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
    filename = (
        time + f"_{kwargs['store_type']}_result_"
        f"{binascii.b2a_hex(os.urandom(8)).decode()}.h5"
    )
    return filename
