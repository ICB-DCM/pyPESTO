"""Optimization result."""

import logging
import warnings
from collections import Counter
from copy import deepcopy
from typing import Sequence, Union

import numpy as np
import pandas as pd

from ..history import HistoryBase
from ..problem import Problem
from ..util import assign_clusters, delete_nan_inf

OptimizationResult = Union['OptimizerResult', 'OptimizeResult']
logger = logging.getLogger(__name__)


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
    optimizer: str
        The optimizer used for optimization.

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
        history: HistoryBase = None,
        exitflag: int = None,
        time: float = None,
        message: str = None,
        optimizer: str = None,
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
        self.history: HistoryBase = history
        self.exitflag: int = exitflag
        self.time: float = time
        self.message: str = message
        self.optimizer = optimizer
        self.free_indices = None

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def summary(self, full: bool = False, show_hess: bool = True) -> str:
        """
        Get summary of the object.

        Parameters
        ----------
        full:
            If True, print full vectors including fixed parameters.
        show_hess:
            If True, display the Hessian of the result.

        Returns
        -------
        summary: str
        """
        # add warning, if self.free_indices is None
        if self.free_indices is None:
            if full:
                logger.warning(
                    "There is no information about fixed parameters, "
                    "run update_to_full with the corresponding problem first."
                )
            full = True
        message = (
            "### Optimizer Result\n\n"
            f"* optimizer used: {self.optimizer}\n"
            f"* message: {self.message} \n"
            f"* number of evaluations: {self.n_fval}\n"
            f"* time taken to optimize: {self.time:0.3f}s\n"
            f"* startpoint: {self.x0 if full else self.x0[self.free_indices]}\n"
            f"* endpoint: {self.x if full else self.x[self.free_indices]}\n"
        )
        # add fval, gradient, hessian, res, sres if available
        if self.fval is not None:
            message += f"* final objective value: {self.fval}\n"
        if self.grad is not None:
            message += (
                f"* final gradient value: "
                f"{self.grad if full else self.grad[self.free_indices]}\n"
            )
        if self.hess is not None and show_hess:
            hess = self.hess
            if not full:
                hess = self.hess[np.ix_(self.free_indices, self.free_indices)]
            message += f"* final hessian value: {hess}\n"
        if self.res is not None:
            message += f"* final residual value: {self.res}\n"
        if self.sres is not None:
            message += f"* final residual sensitivity: {self.sres}\n"

        return message

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
        self.free_indices = np.array(problem.x_free_indices)


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

    def __getstate__(self):
        # while we override __getattr__ as we do now, this is required to keep
        # instances pickle-able
        return vars(self)

    def __setstate__(self, state):
        # while we override __getattr__ as we do now, this is required to keep
        # instances pickle-able
        vars(self).update(state)

    def __len__(self):
        return len(self.list)

    def summary(
        self,
        disp_best: bool = True,
        disp_worst: bool = False,
        full: bool = False,
        show_hess: bool = True,
    ) -> str:
        """
        Get summary of the object.

        Parameters
        ----------
        disp_best:
            Whether to display a detailed summary of the best run.
        disp_worst:
            Whether to display a detailed summary of the worst run.
        full:
            If True, print full vectors including fixed parameters.
        show_hess:
            If True, display the Hessian of the OptimizerResult.
        """
        if len(self) == 0:
            return "## Optimization Result \n\n*empty*\n"

        # perform clustering for better information
        clust, clustsize = assign_clusters(delete_nan_inf(self.fval)[1])

        # aggregate exit messages
        message_counts_df = pd.DataFrame(
            Counter(self.message).most_common(), columns=["Message", "Count"]
        )
        counter_message = message_counts_df[["Count", "Message"]].to_markdown(
            index=False
        )
        counter_message = "  " + counter_message.replace("\n", "\n  ")

        times_message = (
            f'\t* Mean execution time: {np.mean(self.time):0.3f}s\n'
            f'\t* Maximum execution time: {np.max(self.time):0.3f}s,'
            f'\tid={self[np.argmax(self.time)].id}\n'
            f'\t* Minimum execution time: {np.min(self.time):0.3f}s,\t'
            f'id={self[np.argmin(self.time)].id}'
        )

        # special handling in case there are only non-finite fvals
        num_best_value = int(clustsize[0]) if len(clustsize) else len(self)
        num_plateaus = (
            (1 + max(clust) - sum(clustsize == 1)) if len(clustsize) else 0
        )

        summary = (
            "## Optimization Result \n\n"
            f"* number of starts: {len(self)} \n"
            f"* best value: {self[0]['fval']}, id={self[0]['id']}\n"
            f"* worst value: {self[-1]['fval']}, id={self[-1]['id']}\n"
            f"* number of non-finite values: "
            f"{np.logical_not(np.isfinite(self.fval)).sum()}\n\n"
            f"* execution time summary:\n{times_message}\n"
            f"* summary of optimizer messages:\n\n{counter_message}\n\n"
            f"* best value found (approximately) {num_best_value} time(s)\n"
            f"* number of plateaus found: {num_plateaus}\n"
        )
        if disp_best:
            summary += (
                f"\nA summary of the best run:\n\n"
                f"{self[0].summary(full, show_hess=show_hess)}"
            )
        if disp_worst:
            summary += (
                f"\nA summary of the worst run:\n\n"
                f"{self[-1].summary(full)}"
            )
        return summary

    def append(
        self,
        optimize_result: OptimizationResult,
        sort: bool = True,
        prefix: str = '',
    ):
        """
        Append an OptimizerResult or an OptimizeResult to the result object.

        Parameters
        ----------
        optimize_result:
            The result of one or more (local) optimizer run.
        sort:
            Boolean used so we only sort once when appending an
            optimize_result.
        prefix:
            The IDs for all appended results will be prefixed with this.
        """
        current_ids = set(self.id)
        if isinstance(optimize_result, OptimizeResult):
            new_ids = {
                prefix + identifier
                for identifier in optimize_result.id
                if identifier is not None
            }
            if not current_ids.isdisjoint(new_ids):
                raise ValueError(
                    "Some id's you want to merge coincide with "
                    f"the existing id's: {current_ids & new_ids}. "
                    "Please use an appropriate prefix such as 'run_2_'."
                )
            for optimizer_result in optimize_result.list:
                self.append(optimizer_result, sort=False, prefix=prefix)
        elif isinstance(optimize_result, OptimizerResult):
            # if id is None, append without checking for duplicate ids
            if optimize_result.id is None:
                self.list.append(optimize_result)
            else:
                new_id = prefix + optimize_result.id
                if new_id in current_ids:
                    raise ValueError(
                        f"The id `{new_id}` you want to merge coincides with "
                        "the existing id's. Please use an "
                        "appropriate prefix such as 'run_2_'."
                    )
                optimize_result.id = new_id
                self.list.append(optimize_result)
        else:
            raise ValueError(
                "Argument `optimize_result` is of unsupported "
                f"type {type(optimize_result)}."
            )
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
            "optimize_result.key and will be removed in future "
            "releases."
        )
        return [res[key] for res in self.list]

    def get_by_id(self, ores_id: str):
        """Get OptimizationResult with the specified id."""
        for res in self.list:
            if res.id == ores_id:
                return res
        else:
            raise ValueError(f"no optimization result with id={ores_id}")
