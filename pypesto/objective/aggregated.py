import numpy as np

from copy import deepcopy
from typing import Sequence, Dict
from .objective import Objective, ResultDict

from .constants import RDATAS, FVAL, CHI2, SCHI2, RES, SRES, GRAD, HESS, HESSP


class AggregatedObjective(Objective):
    """
    This class aggregates multiple objectives into one objective.
    """

    def __init__(
            self,
            objectives: Sequence[Objective],
            x_names: Sequence[str] = None):
        """
        Constructor.

        Parameters
        ----------
        objectives:
            Sequence of pypesto.objetive instances
        """
        # input typechecks
        if not isinstance(objectives, Sequence):
            raise TypeError(f'Objectives must be a Sequence, '
                            f'was {type(objectives)}.')

        if not all(
                isinstance(objective, Objective)
                for objective in objectives
        ):
            raise TypeError('Objectives must only contain elements of type'
                            'pypesto.Objective')

        if not objectives:
            raise ValueError('Length of objectives must be at least one')

        self._objectives = objectives

        super().__init__(x_names=x_names)

    def __deepcopy__(self, memodict=None):
        other = AggregatedObjective(
            objectives=[deepcopy(objective) for objective in self._objectives],
            x_names=deepcopy(self.x_names),
        )
        return other

    def check_mode(self, mode) -> bool:
        return all(
            objective.check_mode(mode)
            for objective in self._objectives
        )

    def check_sensi_orders(self, sensi_orders, mode) -> bool:
        return all(
            objective.check_sensi_orders(sensi_orders, mode)
            for objective in self._objectives
        )

    def call_unprocessed(self, x, sensi_orders, mode) -> Dict:
        return aggregate_results([
            objective.call_unprocessed(x, sensi_orders, mode)
            for objective in self._objectives
        ])

    def reset_steadystate_guesses(self):
        """
        Propagates reset_steadystate_guesses() to child objectives if available
        (currently only applies for amici_objective)
        """
        for objective in self._objectives:
            if hasattr(objective, 'reset_steadystate_guesses'):
                objective.reset_steadystate_guesses()


def aggregate_results(rvals: Sequence[ResultDict]) -> ResultDict:

    """
    Aggregrate the results from the provided sequence of ResultDicts into a
    single ResultDict. Format of ResultDict is defined in

    Parameters
    ----------
    rvals:
        results to aggregate
    """

    # rvals are guaranteed to be consistent as _check_sensi_orders checks whether 
    # each objective can be called with the respective sensi_orders/mode

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

    # transform results to dict
    if res is not None:
        result[RES] = res
    if sres is not None:
        result[SRES] = sres

    return result
