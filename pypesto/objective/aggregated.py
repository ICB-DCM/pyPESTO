from copy import deepcopy
from typing import List, Dict
from .objective import Objective

from .constants import RDATAS


class AggregatedObjective(Objective):
    """
    This class aggregates multiple objectives into one objective.
    """

    def __init__(
            self,
            objectives: List[Objective],
            x_names: List[str] = None):
        """
        Constructor.

        Parameters
        ----------
        objectives: list
            List of pypesto.objetive instances
        """
        # input typechecks
        if not isinstance(objectives, list):
            raise TypeError(f'Objectives must be a list, '
                            f'was {type(objectives)}.')

        if not all(
                isinstance(objective, Objective)
                for objective in objectives
        ):
            raise TypeError('Objectives must only contain elements of type'
                            'pypesto.Objective')

        if not len(objectives):
            raise ValueError('Length of objectives must be at least one')

        self._objectives = objectives

        super().__init__(x_names=x_names)

    def __deepcopy__(self, memodict=None):
        other = AggregatedObjective(
            objectives=[deepcopy(objective) for objective in self._objectives],
            x_names=deepcopy(self.x_names),
        )
        return other

    def _check_sensi_orders(self, sensi_orders, mode) -> None:
        for objective in self._objectives:
            objective._check_sensi_orders(sensi_orders, mode)

    def call_unprocessed(self, x, sensi_orders, mode) -> Dict:
        return sum(
            objective.call_unprocessed(x, sensi_orders, mode)
            for objective in self._objectives
        )

    def reset_steadystate_guesses(self):
        """
        Propagates reset_steadystate_guesses() to child objectives if available
        (currently only applies for amici_objective)
        """
        for objective in self._objectives:
            if hasattr(objective, 'reset_steadystate_guesses'):
                objective.reset_steadystate_guesses()
