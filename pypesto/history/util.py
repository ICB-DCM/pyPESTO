"""History utility functions."""

import numbers
from collections.abc import Sequence
from functools import wraps
from typing import Union

import numpy as np

from ..C import SUFFIXES

ResultDict = dict[str, Union[float, np.ndarray]]
MaybeArray = Union[np.ndarray, "np.nan"]


class HistoryTypeError(ValueError):
    """Error raised when an unsupported history type is requested."""

    def __init__(self, history_type: str):
        super().__init__(
            f"Unsupported history type: {history_type}, expected {SUFFIXES}"
        )


class CsvHistoryTemplateError(ValueError):
    """Error raised when no template is given for CSV history."""

    def __init__(self, storage_file: str):
        super().__init__(
            "CSV History requires an `{id}` template in the `storage_file`, "
            f"but is {storage_file}"
        )


def trace_wrap(f):
    """
    Wrap around trace getters.

    Transform input `ix` vectors to a valid index list, and reduce for
    integer `ix` the output to a single value.
    """

    @wraps(f)
    def wrapped_f(
        self, ix: Union[Sequence[int], int, None] = None, trim: bool = False
    ) -> Union[Sequence[Union[float, MaybeArray]], Union[float, MaybeArray]]:
        # whether to reduce the output
        reduce = isinstance(ix, numbers.Integral)
        # default: full list
        if ix is None:
            if trim:
                ix = self.get_trimmed_indices()
            else:
                ix = np.arange(0, len(self), dtype=int)
        # turn every input into an index list
        if reduce:
            ix = np.array([ix], dtype=int)
        # obtain the trace
        trace = f(self, ix)
        # reduce the output
        if reduce:
            trace = trace[0]
        return trace

    return wrapped_f
