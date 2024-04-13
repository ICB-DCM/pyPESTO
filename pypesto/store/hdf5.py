"""Convenience functions for working with HDF5 files."""

from collections.abc import Collection
from numbers import Integral, Number, Real

import h5py
import numpy as np


def write_array(f: h5py.Group, path: str, values: Collection) -> None:
    """
    Write array to hdf5.

    Parameters
    ----------
    f:
        h5py.Group  where dataset should be created
    path:
        path of the dataset to create
    values:
        array to write
    """
    if all(isinstance(x, Integral) for x in values):
        write_int_array(f, path, values)
    elif all(isinstance(x, Real) for x in values):
        write_float_array(f, path, values)
    elif all(isinstance(x, str) for x in values):
        write_string_array(f, path, values)
    else:
        f[path] = values


def write_string_array(f: h5py.Group, path: str, strings: Collection) -> None:
    """
    Write string array to hdf5.

    Parameters
    ----------
    f:
        h5py.Group where dataset should be created
    path:
        path of the dataset to create
    strings:
        list of strings to be written to f
    """
    dt = h5py.special_dtype(vlen=str)
    dset = f.create_dataset(path, (len(strings),), dtype=dt)

    if len(strings):
        dset[:] = [s.encode("utf8") for s in strings]


def write_float_array(
    f: h5py.Group, path: str, values: Collection[Number], dtype="f8"
) -> None:
    """
    Write float array to hdf5.

    Parameters
    ----------
    f:
        h5py.Group where dataset should be created
    path:
        path of the dataset to create
    values:
        array to write
    dtype:
        datatype
    """
    if path not in f:
        dset = f.create_dataset(path, (np.shape(values)), dtype=dtype)
    else:
        dset = f[path]

    if len(values):
        dset[:] = values


def write_int_array(
    f: h5py.Group, path: str, values: Collection[int], dtype="<i4"
):
    """
    Write integer array to hdf5.

    Parameters
    ----------
    f:
        h5py.Group where dataset should be created
    path:
        path of the dataset to create
    values:
        array to write
    dtype:
        datatype
    """
    dset = f.create_dataset(path, (len(values),), dtype=dtype)

    if len(values):
        dset[:] = values
