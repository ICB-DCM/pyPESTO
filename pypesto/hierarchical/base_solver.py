from typing import Union

import numpy as np

from .base_problem import InnerProblem


class InnerSolver:
    """Solver for an inner optimization problem."""

    def initialize(self):
        """
        (Re-)initialize the solver.

        Default: Do nothing.
        """

    def solve(
        self,
        problem: InnerProblem,
        sim: list[np.ndarray],
        sigma: list[np.ndarray],
        scaled: bool,
    ) -> Union[dict[str, float], list]:
        """Solve the subproblem.

        Parameters
        ----------
        problem:
            The inner problem to solve.
        sim:
            List of model output matrices, as provided in AMICI's
            ``ReturnData.y``. Same order as simulations in the
            PEtab problem.
        sigma:
            List of sigma matrices from the model, as provided in AMICI's
            ``ReturnData.sigmay``. Same order as simulations in the
            PEtab problem.
        scaled:
            Whether to scale the results to the parameter scale specified in
            ``problem``.

        Returns
        -------
        A dictionary of inner parameter ids and their optimal values for the
        relative inner problem, or a list of inner optimization results
        for the semiquantitative and ordinal inner problems.
        """
