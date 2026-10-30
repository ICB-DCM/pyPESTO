import inspect
import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from ..C import (
    FVAL,
    GRAD,
    HESS,
    HESSP,
    RDATAS,
    RES,
    SRES,
    ModeType,
)
from .base import ObjectiveBase, ResultDict


class AggregatedObjective(ObjectiveBase):
    """Aggregates multiple objectives into one objective."""

    def __init__(
        self,
        objectives: Sequence[ObjectiveBase],
        x_names: Sequence[str] = None,
        x_indices: Sequence[Sequence[int] | None] | None = None,
    ):
        """
        Initialize objective.

        Parameters
        ----------
        objectives:
            Sequence of pypesto.ObjectiveBase instances
        x_names:
            Sequence of names of the (optimized) parameters.
            (Details see documentation of x_names in
            :class:`pypesto.ObjectiveBase`)
        x_indices:
            For each objective, the indices (in ``x_names``) of the parameters
            it takes, in the order it takes them. Its derivatives are mapped
            back to the full parameter vector; those w.r.t. the other
            parameters are zero. An entry ``None`` stands for all parameters.
            If ``x_indices`` is ``None`` (the default), every objective takes
            the full parameter vector. Requires ``x_names``.
        """
        # input typechecks
        if not isinstance(objectives, Sequence):
            raise TypeError(
                f"Objectives must be a Sequence, was {type(objectives)}."
            )

        if not all(
            isinstance(objective, ObjectiveBase) for objective in objectives
        ):
            raise TypeError(
                "Objectives must only contain elements of type"
                "pypesto.Objective"
            )

        if not objectives:
            raise ValueError("Length of objectives must be at least one")

        self._objectives = objectives
        self._x_indices = (
            None
            if x_indices is None
            else _check_x_indices(objectives, x_names, x_indices)
        )

        super().__init__(x_names=x_names)

    def __deepcopy__(self, memodict=None):
        """Create copy of objective."""
        other = AggregatedObjective(
            objectives=[deepcopy(objective) for objective in self._objectives],
            x_names=deepcopy(self.x_names),
            x_indices=deepcopy(self._x_indices),
        )
        for key in set(self.__dict__.keys()) - {
            "_objectives",
            "x_names",
            "_x_indices",
        }:
            other.__dict__[key] = deepcopy(self.__dict__[key])

        return other

    def check_mode(self, mode: ModeType) -> bool:
        """See `ObjectiveBase` documentation."""
        return all(
            objective.check_mode(mode) for objective in self._objectives
        )

    def check_sensi_orders(
        self,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
    ) -> bool:
        """See `ObjectiveBase` documentation."""
        return all(
            objective.check_sensi_orders(sensi_orders, mode)
            for objective in self._objectives
        )

    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        kwargs_list: Sequence[dict[str, Any]] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> ResultDict:
        """
        See `ObjectiveBase` for more documentation.

        Main method to overwrite from the base class. It handles and
        delegates the actual objective evaluation.

        Parameters
        ----------
        kwargs_list:
            Objective-specific keyword arguments, where the dictionaries are
            ordered by the objectives.
        """
        if kwargs_list is None:
            kwargs_list = [{}] * len(self._objectives)
        elif len(kwargs_list) != len(self._objectives):
            raise ValueError(
                "The length of `kwargs_list` must match the number of "
                "objectives you are aggregating."
            )
        for objective_, objective_kwargs in zip(
            self._objectives, kwargs_list, strict=True
        ):
            if (
                "return_dict"
                in inspect.signature(objective_.call_unprocessed).parameters
            ):
                objective_kwargs["return_dict"] = return_dict
            else:
                warnings.warn(
                    "Please add `return_dict` to the argument list of your "
                    "objective's `call_unprocessed` method. "
                    f"Current objective: `{type(objective_)}`.",
                    DeprecationWarning,
                    stacklevel=1,
                )
        return aggregate_results(
            [
                _expand_result(
                    objective.call_unprocessed(
                        x if x_ix is None else x[x_ix],
                        sensi_orders,
                        mode,
                        **kwargs,
                        **cur_kwargs,
                    ),
                    x_ix=x_ix,
                    dim=len(x),
                )
                for objective, x_ix, cur_kwargs in zip(
                    self._objectives,
                    self._x_indices or [None] * len(self._objectives),
                    kwargs_list,
                    strict=True,
                )
            ]
        )

    def initialize(self):
        """See `ObjectiveBase` documentation."""
        for objective in self._objectives:
            objective.initialize()

    def get_config(self) -> dict:
        """Return basic information of the objective configuration."""
        info = super().get_config()
        for n_obj, obj in enumerate(self._objectives):
            info[f"objective_{n_obj}"] = obj.get_config()
        return info


def _check_x_indices(
    objectives: Sequence[ObjectiveBase],
    x_names: Sequence[str] | None,
    x_indices: Sequence[Sequence[int] | None],
) -> list[np.ndarray]:
    """Check ``x_indices`` and replace ``None`` by all parameter indices.

    See :class:`AggregatedObjective`.
    """
    if x_names is None:
        raise ValueError("`x_indices` requires `x_names`.")
    if len(x_indices) != len(objectives):
        raise ValueError(
            "The length of `x_indices` must match the number of objectives."
        )
    dim = len(x_names)
    checked = []
    for objective, x_ix in zip(objectives, x_indices, strict=True):
        x_ix = np.arange(dim) if x_ix is None else np.asarray(x_ix, dtype=int)
        if x_ix.ndim != 1 or len(set(x_ix)) != len(x_ix):
            raise ValueError(f"`x_indices` must be unique, got {x_ix}.")
        if np.any((x_ix < 0) | (x_ix >= dim)):
            raise ValueError(f"`x_indices` must be in [0, {dim}), got {x_ix}.")
        # objectives without own parameter names cannot be checked further
        if objective.x_names is not None and list(objective.x_names) != [
            x_names[ix] for ix in x_ix
        ]:
            raise ValueError(
                f"`x_indices` select {[x_names[ix] for ix in x_ix]}, but "
                f"the objective takes {list(objective.x_names)}."
            )
        checked.append(x_ix)
    return checked


def _expand_result(
    rval: ResultDict, x_ix: np.ndarray | None, dim: int
) -> ResultDict:
    """Map the derivatives w.r.t. the parameters ``x_ix`` to all parameters.

    The objective does not depend on the other parameters.
    """
    if x_ix is None:
        return rval
    rval = dict(rval)
    if (grad := rval.get(GRAD)) is not None:
        rval[GRAD] = np.zeros(dim)
        rval[GRAD][x_ix] = grad
    if (hess := rval.get(HESS)) is not None:
        rval[HESS] = np.zeros((dim, dim))
        rval[HESS][np.ix_(x_ix, x_ix)] = hess
    if (sres := rval.get(SRES)) is not None:
        sres = np.asarray(sres)
        rval[SRES] = np.zeros((sres.shape[0], dim))
        rval[SRES][:, x_ix] = sres
    return rval


def aggregate_results(rvals: Sequence[ResultDict]) -> ResultDict:
    """
    Aggregate the results from the provided ResultDicts into a single one.

    Parameters
    ----------
    rvals:
        results to aggregate
    """
    # sum over fval/grad/hess, if available in all rvals
    result = {
        key: sum(rval[key] for rval in rvals)
        for key in [FVAL, GRAD, HESS, HESSP]
        if all(key in rval for rval in rvals)
    }

    # extract rdatas and flatten
    result[RDATAS] = []
    for rval in rvals:
        if RDATAS in rval:
            result[RDATAS].extend(rval[RDATAS])

    # initialize res and sres
    if RES in rvals[0]:
        res = np.asarray(rvals[0][RES])
    else:
        res = None

    if SRES in rvals[0]:
        sres = np.asarray(rvals[0][SRES])
    else:
        sres = None

    # skip iobj=0 after initialization, stack matrices
    for rval in rvals[1:]:
        if res is not None:
            res = np.hstack([res, np.asarray(rval[RES])])
        if sres is not None:
            sres = np.vstack([sres, np.asarray(rval[SRES])])

    # fill res, sres into result
    if res is not None:
        result[RES] = res
    if sres is not None:
        result[SRES] = sres

    return result
