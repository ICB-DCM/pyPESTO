"""Optimization result."""

import warnings
from copy import deepcopy
from typing import Sequence, Union

import numpy as np
import pandas as pd

from ..objective import History
from ..problem import Problem

OptimizationResult = Union['OptimizerResult', 'OptimizeResult']


class OptimizerResult(dict):
    """
    The result of an optimizer run.

    Used as a standardized return value to map from the individual result
    objects returned by the employed optimizers to the format understood by
    pypesto.

    Can be used like a dict.

    Attributes
    ----------
    id:
        Id of the optimizer run. Usually the start index.
    x:
        The best found parameters.
    fval:
        The best found function value, `fun(x)`.
    grad:
        The gradient at `x`.
    hess:
        The Hessian at `x`.
    res:
        The residuals at `x`.
    sres:
        The residual sensitivities at `x`.
    n_fval
        Number of function evaluations.
    n_grad:
        Number of gradient evaluations.
    n_hess:
        Number of Hessian evaluations.
    n_res:
        Number of residuals evaluations.
    n_sres:
        Number of residual sensitivity evaluations.
    x0:
        The starting parameters.
    fval0:
        The starting function value, `fun(x0)`.
    history:
        Objective history.
    exitflag:
        The exitflag of the optimizer.
    time:
        Execution time.
    message: str
        Textual comment on the optimization result.

    Notes
    -----
    Any field not supported by the optimizer is filled with None.
    """

    def __init__(
        self,
        id: str = None,
        x: np.ndarray = None,
        fval: float = None,
        grad: np.ndarray = None,
        hess: np.ndarray = None,
        res: np.ndarray = None,
        sres: np.ndarray = None,
        n_fval: int = None,
        n_grad: int = None,
        n_hess: int = None,
        n_res: int = None,
        n_sres: int = None,
        x0: np.ndarray = None,
        fval0: float = None,
        history: History = None,
        exitflag: int = None,
        time: float = None,
        message: str = None,
    ):
        super().__init__()
        self.id = id
        self.x: np.ndarray = np.array(x) if x is not None else None
        self.fval: float = fval
        self.grad: np.ndarray = np.array(grad) if grad is not None else None
        self.hess: np.ndarray = np.array(hess) if hess is not None else None
        self.res: np.ndarray = np.array(res) if res is not None else None
        self.sres: np.ndarray = np.array(sres) if sres is not None else None
        self.n_fval: int = n_fval
        self.n_grad: int = n_grad
        self.n_hess: int = n_hess
        self.n_res: int = n_res
        self.n_sres: int = n_sres
        self.x0: np.ndarray = np.array(x0) if x0 is not None else None
        self.fval0: float = fval0
        self.history: History = history
        self.exitflag: int = exitflag
        self.time: float = time
        self.message: str = message

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def update_to_full(self, problem: Problem) -> None:
        """
        Update values to full vectors/matrices.

        Parameters
        ----------
        problem:
            problem which contains info about how to convert to full vectors
            or matrices
        """
        self.x = problem.get_full_vector(self.x, problem.x_fixed_vals)
        self.grad = problem.get_full_vector(self.grad)
        self.hess = problem.get_full_matrix(self.hess)
        self.x0 = problem.get_full_vector(self.x0, problem.x_fixed_vals)


class OptimizeResult:
    """Result of the :py:func:`pypesto.optimize.minimize` function."""

    def __init__(self):
        self.list = []

    def __deepcopy__(self, memo):
        other = OptimizeResult()
        other.list = deepcopy(self.list)
        return other

    def __getattr__(self, key):
        """Define `optimize_result.key`."""
        try:
            return [res[key] for res in self.list]
        except KeyError:
            raise AttributeError(key)

    def __getitem__(self, index):
        """Define `optimize_result[i]` to access the i-th result."""
        try:
            return self.list[index]
        except IndexError:
            raise IndexError(
                f"{index} out of range for optimize result of "
                f"length {len(self.list)}."
            )

    def __len__(self):
        return len(self.list)

    def append(self, opt_result: OptimizationResult, sort: bool = True):
        """
        Append an OptimizerResult or an OptimizeResult to the result object.

        Parameters
        ----------
        opt_result:
            The result of one or more (local) optimizer run.
        sort:
            Boolean used so we only sort once when appending an
            optimize_result.
        """
        if isinstance(opt_result, OptimizeResult):
            for optimizer_result in opt_result.list:
                self.append(optimizer_result, sort=False)
        elif isinstance(opt_result, OptimizerResult):
            opt_result.id = self._assign_unique_id(opt_result.id)
            self.list.append(opt_result)
        if sort:
            self.sort()

    def sort(self):
        """Sort the optimizer results by function value fval (ascending)."""

        def get_fval(res):
            return res.fval if not np.isnan(res.fval) else np.inf

        self.list = sorted(self.list, key=get_fval)

    def as_dataframe(self, keys=None) -> pd.DataFrame:
        """
        Get as pandas DataFrame.

        If keys is a list, return only the specified values, otherwise all.
        """
        lst = self.as_list(keys)

        df = pd.DataFrame(lst)

        return df

    def as_list(self, keys=None) -> Sequence:
        """
        Get as list.

        If keys is a list, return only the specified values.

        Parameters
        ----------
        keys: list(str), optional
            Labels of the field to extract.
        """
        lst = self.list

        if keys is not None:
            lst = [{key: res[key] for key in keys} for res in lst]

        return lst

    def get_for_key(self, key) -> list:
        """Extract the list of values for the specified key as a list."""
        warnings.warn(
            "get_for_key() is deprecated in favour of "
            "optimize_result['key'] and will be removed in future "
            "releases."
        )
        return [res[key] for res in self.list]

    def _assign_unique_id(self, id: str, suffix: int = None):
        """
        Assign a unique id to an id if needed.

        Returns id if not used already otherwise tries id_suffix with
        increasing suffix until it is unique.
        """
        if suffix is None:
            if id not in self.id:
                return id
            return self._assign_unique_id(id=id, suffix=1)
        if f'{id}_{suffix}' not in self.id:
            return f'{id}_{suffix}'
        return self._assign_unique_id(id=id, suffix=suffix + 1)
