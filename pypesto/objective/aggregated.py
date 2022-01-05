from copy import deepcopy
from typing import Sequence, Tuple

import numpy as np

from ..C import CHI2, FVAL, GRAD, HESS, HESSP, RDATAS, RES, SCHI2, SRES
from .base import ObjectiveBase, ResultDict


class AggregatedObjective(ObjectiveBase):
    """Aggregates multiple objectives into one objective."""

    def __init__(
        self,
        objectives: Sequence[ObjectiveBase],
        x_names: Sequence[str] = None,
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
        """
        # input typechecks
        if not isinstance(objectives, Sequence):
            raise TypeError(f'Objectives must be a Sequence, '
                            f'was {type(objectives)}.')

        if not all(
                isinstance(objective, ObjectiveBase)
                for objective in objectives
        ):
            raise TypeError('Objectives must only contain elements of type'
                            'pypesto.Objective')

        if not objectives:
            raise ValueError('Length of objectives must be at least one')

        self._objectives = objectives

        super().__init__(x_names=x_names)

    def __deepcopy__(self, memodict=None):
        """Create copy of objective."""
        other = AggregatedObjective(
            objectives=[deepcopy(objective) for objective in self._objectives],
            x_names=deepcopy(self.x_names),
        )
        for key in set(self.__dict__.keys()) - {'_objectives', 'x_names'}:
            other.__dict__[key] = deepcopy(self.__dict__[key])

        return other

    def check_mode(self, mode: str) -> bool:
        """See `ObjectiveBase` documentation."""
        return all(
            objective.check_mode(mode)
            for objective in self._objectives
        )

    def check_sensi_orders(
        self,
        sensi_orders: Tuple[int, ...],
        mode: str,
    ) -> bool:
        """See `ObjectiveBase` documentation."""
        return all(
            objective.check_sensi_orders(sensi_orders, mode)
            for objective in self._objectives
        )

    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: str,
        **kwargs,
    ) -> ResultDict:
        """
        See `ObjectiveBase` for more documentation.

        Main method to overwrite from the base class. It handles and
        delegates the actual objective evaluation.
        """
        return aggregate_results([
            objective.call_unprocessed(x, sensi_orders, mode, **kwargs)
            for objective in self._objectives
        ])

    def initialize(self):
        """See `ObjectiveBase` documentation."""
        for objective in self._objectives:
            objective.initialize()

    def get_config(self) -> dict:
        """Return basic information of the objective configuration."""
        info = super().get_config()
        for n_obj, obj in enumerate(self._objectives):
            info[f'objective_{n_obj}'] = obj.get_config()
        return info


def aggregate_results(rvals: Sequence[ResultDict]) -> ResultDict:
    """
    Aggregrate the results from the provided ResultDicts into a single one.

    Parameters
    ----------
    rvals:
        results to aggregate
    """
    # rvals are guaranteed to be consistent as _check_sensi_orders checks
    # whether each objective can be called with the respective
    # sensi_orders/mode

    # sum over fval/grad/hess
    result = {
        key: sum(rval[key] for rval in rvals)
        for key in [FVAL, CHI2, SCHI2, GRAD, HESS, HESSP]
        if rvals[0].get(key, None) is not None
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
