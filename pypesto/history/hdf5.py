"""HDF5 history."""


import contextlib
import time
from typing import Dict, Sequence, Tuple, Union

import h5py
import numpy as np

from ..C import (
    EXITFLAG,
    FVAL,
    GRAD,
    HESS,
    HISTORY,
    MESSAGE,
    MESSAGES,
    MODE_FUN,
    MODE_RES,
    N_FVAL,
    N_GRAD,
    N_HESS,
    N_ITERATIONS,
    N_RES,
    N_SRES,
    RES,
    SRES,
    START_TIME,
    TIME,
    TRACE,
    TRACE_SAVE_ITER,
    ModeType,
    X,
)
from .base import HistoryBase, add_fun_from_res, reduce_result_via_options
from .options import HistoryOptions
from .util import MaybeArray, ResultDict, trace_wrap


def with_h5_file(mode: str):
    """Wrap function to work with hdf5 file.

    Parameters
    ----------
    mode:
        Access mode, see
        https://docs.h5py.org/en/stable/high/file.html.
    """
    modes = ["r", "a"]
    if mode not in modes:
        # can be extended if reasonable
        raise ValueError(f"Mode must be one of {modes}")

    def decorator(fun):
        def wrapper(self, *args, **kwargs):
            # file already opened
            if self._f is not None and (
                mode == self._f.mode
                or mode == "r"
                or (self._f.mode == "r+" and mode == "a")
            ):
                return fun(self, *args, **kwargs)

            with h5py.File(self.file, mode) as f:
                self._f = f
                ret = fun(self, *args, **kwargs)
                self._f = None
                return ret

        return wrapper

    return decorator


def check_editable(fun):
    """Check if the history is editable."""

    def wrapper(self, *args, **kwargs):
        if not self.editable:
            raise ValueError(
                f'ID "{self.id}" is already used in history file '
                f'"{self.file}".'
            )
        return fun(self, *args, **kwargs)

    return wrapper


class Hdf5History(HistoryBase):
    """
    Stores a representation of the history in an HDF5 file.

    Parameters
    ----------
    id:
        Id of the history
    file:
        HDF5 file name.
    options:
        History options.
    """

    def __init__(
        self,
        id: str,
        file: str,
        options: Union[HistoryOptions, Dict] = None,
    ):
        super().__init__(options=options)
        self.id: str = id
        self.file: str = file

        # filled during file access
        self._f: Union[h5py.File, None] = None

        # to check whether the trace can be edited
        self.editable: bool = self._editable()

    @check_editable
    @with_h5_file("a")
    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        """See `History` docstring."""
        # check whether the file was marked as editable upon initialization
        super().update(x, sensi_orders, mode, result)
        self._update_counts(sensi_orders, mode)
        self._update_trace(x, sensi_orders, mode, result)

    @with_h5_file("a")
    @check_editable
    def finalize(self, message: str = None, exitflag: str = None) -> None:
        """See `HistoryBase` docstring."""
        super().finalize()

        # add message and exitflag to trace
        f = self._f
        if f'{HISTORY}/{self.id}/{MESSAGES}/' not in f:
            f.create_group(f'{HISTORY}/{self.id}/{MESSAGES}/')
        grp = f[f'{HISTORY}/{self.id}/{MESSAGES}/']
        if message is not None:
            grp.attrs[MESSAGE] = message
        if exitflag is not None:
            grp.attrs[EXITFLAG] = exitflag

    @staticmethod
    def load(
        id: str, file: str, options: Union[HistoryOptions, Dict] = None
    ) -> 'Hdf5History':
        """Load the History object from memory."""
        history = Hdf5History(id=id, file=file, options=options)
        if options is None:
            history.recover_options(file)
        return history

    def recover_options(self, file: str):
        """Recover options when loading the hdf5 history from memory.

        Done by testing which entries were recorded.
        """
        trace_record = self._has_non_nan_entries(X)
        trace_record_grad = self._has_non_nan_entries(GRAD)
        trace_record_hess = self._has_non_nan_entries(HESS)
        trace_record_res = self._has_non_nan_entries(RES)
        trace_record_sres = self._has_non_nan_entries(SRES)

        restored_history_options = HistoryOptions(
            trace_record=trace_record,
            trace_record_grad=trace_record_grad,
            trace_record_hess=trace_record_hess,
            trace_record_res=trace_record_res,
            trace_record_sres=trace_record_sres,
            trace_save_iter=self.trace_save_iter,
            storage_file=file,
        )

        self.options = restored_history_options

    def _has_non_nan_entries(self, hdf5_group: str) -> bool:
        """Check if there exist non-nan entries stored for a given group."""
        group = self._get_hdf5_entries(hdf5_group, ix=None)

        for entry in group:
            if not (entry is None or np.all(np.isnan(entry))):
                return True

        return False

    @with_h5_file("a")
    def _update_counts(self, sensi_orders: Tuple[int, ...], mode: ModeType):
        """Update the counters in the hdf5."""
        group = self._require_group()

        if mode == MODE_FUN:
            if 0 in sensi_orders:
                group.attrs[N_FVAL] += 1
            if 1 in sensi_orders:
                group.attrs[N_GRAD] += 1
            if 2 in sensi_orders:
                group.attrs[N_HESS] += 1
        elif mode == MODE_RES:
            if 0 in sensi_orders:
                group.attrs[N_RES] += 1
            if 1 in sensi_orders:
                group.attrs[N_SRES] += 1

    @with_h5_file("r")
    def __len__(self) -> int:
        """Define length of history object."""
        try:
            return self._get_group().attrs[N_ITERATIONS]
        except KeyError:
            return 0

    @property
    @with_h5_file("r")
    def n_fval(self) -> int:
        """See `HistoryBase` docstring."""
        try:
            return self._get_group().attrs[N_FVAL]
        except KeyError:
            return 0

    @property
    @with_h5_file("r")
    def n_grad(self) -> int:
        """See `HistoryBase` docstring."""
        try:
            return self._get_group().attrs[N_GRAD]
        except KeyError:
            return 0

    @property
    @with_h5_file("r")
    def n_hess(self) -> int:
        """See `HistoryBase` docstring."""
        try:
            return self._get_group().attrs[N_HESS]
        except KeyError:
            return 0

    @property
    @with_h5_file("r")
    def n_res(self) -> int:
        """See `HistoryBase` docstring."""
        try:
            return self._get_group().attrs[N_RES]
        except KeyError:
            return 0

    @property
    @with_h5_file("r")
    def n_sres(self) -> int:
        """See `HistoryBase` docstring."""
        try:
            return self._get_group().attrs[N_SRES]
        except KeyError:
            return 0

    @property
    @with_h5_file("r")
    def trace_save_iter(self) -> int:
        """After how many iterations to store the trace."""
        try:
            return self._get_group().attrs[TRACE_SAVE_ITER]
        except KeyError:
            return 0

    @property
    @with_h5_file("r")
    def start_time(self) -> float:
        """See `HistoryBase` docstring."""
        # TODO Y This should also be saved in and recovered from the hdf5 file
        try:
            return self._get_group().attrs[START_TIME]
        except KeyError:
            return np.nan

    @property
    @with_h5_file("r")
    def message(self) -> str:
        """Optimizer message in case of finished optimization."""
        try:
            return self._f[f'{HISTORY}/{self.id}/{MESSAGES}/'].attrs[MESSAGE]
        except KeyError:
            return None

    @property
    @with_h5_file("r")
    def exitflag(self) -> str:
        """Optimizer exitflag in case of finished optimization."""
        try:
            return self._f[f'{HISTORY}/{self.id}/{MESSAGES}/'].attrs[EXITFLAG]
        except KeyError:
            return None

    @with_h5_file("a")
    def _update_trace(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        """Update and possibly store the trace."""
        if not self.options.trace_record:
            return

        # calculating function values from residuals
        #  and reduce via requested history options
        result = reduce_result_via_options(
            add_fun_from_res(result), self.options
        )

        used_time = time.time() - self.start_time

        values = {
            X: x,
            FVAL: result[FVAL],
            GRAD: result[GRAD],
            RES: result[RES],
            SRES: result[SRES],
            HESS: result[HESS],
            TIME: used_time,
        }

        iteration = self._require_group().attrs[N_ITERATIONS]

        for key in values.keys():
            if values[key] is not None:
                self._require_group()[f'{iteration}/{key}'] = values[key]

        self._require_group().attrs[N_ITERATIONS] += 1

    @with_h5_file("r")
    def _get_group(self) -> h5py.Group:
        """Get the HDF5 group for the current history."""
        return self._f[f'{HISTORY}/{self.id}/{TRACE}/']

    @with_h5_file("a")
    def _require_group(self) -> h5py.Group:
        """Get, or if necessary create, the group in the hdf5 file."""
        with contextlib.suppress(KeyError):
            return self._f[f'{HISTORY}/{self.id}/{TRACE}/']

        grp = self._f.create_group(f'{HISTORY}/{self.id}/{TRACE}/')
        grp.attrs[N_ITERATIONS] = 0
        grp.attrs[N_FVAL] = 0
        grp.attrs[N_GRAD] = 0
        grp.attrs[N_HESS] = 0
        grp.attrs[N_RES] = 0
        grp.attrs[N_SRES] = 0
        grp.attrs[START_TIME] = time.time()
        # TODO Y it makes no sense to save this here
        #  Also, we do not seem to evaluate this at all
        grp.attrs[TRACE_SAVE_ITER] = self.options.trace_save_iter
        return grp

    @with_h5_file("r")
    def _get_hdf5_entries(
        self,
        entry_id: str,
        ix: Union[int, Sequence[int], None] = None,
    ) -> Sequence:
        """
        Get entries for field `entry_id` from HDF5 file, for indices `ix`.

        Parameters
        ----------
        entry_id:
            The key whose trace is returned.
        ix:
            Index or list of indices of the iterations that will produce
            the trace.

        Returns
        -------
        The entries ix for the key entry_id.
        """
        if ix is None:
            ix = range(len(self))
        trace_result = []

        for iteration in ix:
            try:
                dataset = self._f[
                    f'{HISTORY}/{self.id}/{TRACE}/{iteration}/{entry_id}'
                ]
                if dataset.shape == ():
                    entry = dataset[()]  # scalar
                else:
                    entry = np.array(dataset)
                trace_result.append(entry)
            except KeyError:
                trace_result.append(None)

        return trace_result

    @trace_wrap
    def get_x_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(X, ix)

    @trace_wrap
    def get_fval_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(FVAL, ix)

    @trace_wrap
    def get_grad_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(GRAD, ix)

    @trace_wrap
    def get_hess_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(HESS, ix)

    @trace_wrap
    def get_res_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(RES, ix)

    @trace_wrap
    def get_sres_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(SRES, ix)

    @trace_wrap
    def get_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(TIME, ix)

    def _editable(self) -> bool:
        """
        Check whether the id is already existent in the file.

        Parameters
        ----------
        file:
            HDF5 file name.

        Returns
        -------
        True if the file is editable, False otherwise.
        """
        try:
            with h5py.File(self.file, "a") as f:
                # editable if the id entry does not exist
                if HISTORY not in f.keys() or self.id not in f[HISTORY]:
                    return True
                return False
        except OSError:
            # if something goes wrong, we assume the file is not editable
            return False
