"""History options."""

from pathlib import Path
from typing import Union

from ..C import SUFFIXES, SUFFIXES_CSV
from .util import CsvHistoryTemplateError, HistoryTypeError


class HistoryOptions(dict):
    """
    Options for what values to record.

    In addition implements a factory pattern to generate history objects.

    Parameters
    ----------
    trace_record:
        Flag indicating whether to record the trace of function calls.
        The trace_record_* flags only become effective if
        trace_record is True. Defaults to ``False``.
    trace_record_grad:
        Flag indicating whether to record the gradient in the trace. Defaults
        to ``True``.
    trace_record_hess:
        Flag indicating whether to record the Hessian in the trace. Defaults
        to ``True``.
    trace_record_res:
        Flag indicating whether to record the residual in
        the trace. Defaults to ``True``.
    trace_record_sres:
        Flag indicating whether to record the residual sensitivities in
        the trace. Defaults to ``True``.
    trace_save_iter:
        After how many iterations to store the trace. Defaults to ``10``.
    storage_file:
        File to save the history to. Can be any of None, a
        "{filename}.csv", or a "{filename}.hdf5" file. Depending on the values,
        the :func:`create_history` method creates the appropriate object.
        Occurrences of "{id}" in the file name are replaced by the `id`
        upon creation of a history, if applicable. Defaults to ``None``.
    """

    def __init__(
        self,
        trace_record: bool = False,
        trace_record_grad: bool = True,
        trace_record_hess: bool = True,
        trace_record_res: bool = True,
        trace_record_sres: bool = True,
        trace_save_iter: int = 10,
        storage_file: Union[str, Path, None] = None,
    ):
        super().__init__()

        self.trace_record: bool = trace_record
        self.trace_record_grad: bool = trace_record_grad
        self.trace_record_hess: bool = trace_record_hess
        self.trace_record_res: bool = trace_record_res
        self.trace_record_sres: bool = trace_record_sres
        self.trace_save_iter: int = trace_save_iter
        self.storage_file: str = storage_file

        self._sanity_check()

    def __getattr__(self, key):
        """Allow to use keys as attributes."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def _sanity_check(self):
        """Apply basic sanity checks."""
        if self.storage_file is None:
            return

        # extract storage type
        suffix = Path(self.storage_file).suffix[1:]

        # check storage format is valid
        if suffix not in SUFFIXES:
            raise HistoryTypeError(suffix)

        # check csv histories are parametrized
        if suffix in SUFFIXES_CSV and "{id}" not in self.storage_file:
            raise CsvHistoryTemplateError(self.storage_file)

    @staticmethod
    def assert_instance(
        maybe_options: Union["HistoryOptions", dict],
    ) -> "HistoryOptions":
        """
        Return a valid options object.

        Parameters
        ----------
        maybe_options: :class:`HistoryOptions` or dict
        """
        if isinstance(maybe_options, HistoryOptions):
            return maybe_options
        options = HistoryOptions(**maybe_options)
        return options
