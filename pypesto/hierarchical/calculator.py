from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import amici
    from amici.parameter_mapping import ParameterMapping
except ImportError:
    pass

from ..C import (
    AMICI_SIGMAY,
    AMICI_Y,
    GRAD,
    HESS,
    INNER_PARAMETERS,
    INNER_RDATAS,
    RDATAS,
    ModeType,
)
from ..objective.amici.amici_calculator import (
    AmiciCalculator,
    AmiciModel,
    AmiciSolver,
)
from .problem import AmiciInnerProblem
from .solver import AnalyticalInnerSolver, InnerSolver

# ToDo: .


class HierarchicalAmiciCalculator(AmiciCalculator):
    """A calculator that is passed as `calculator` to the pypesto.AmiciObjective."""

    def __init__(
        self,
        inner_problem: AmiciInnerProblem,
        inner_solver: Optional[InnerSolver] = None,
    ):
        """Initialize the calculator from the given problem.

        Arguments
        ---------
        inner_problem:
            The inner problem of a hierarchical optimization problem.
        inner_solver:
            A solver to solve ``inner_problem``.
            Defaults to ``pypesto.hierarchical.solver.AnalyticalInnerSolver``.
        """
        super().__init__()

        self.inner_problem = inner_problem

        if inner_solver is None:
            inner_solver = AnalyticalInnerSolver()
        self.inner_solver = inner_solver
        self.simulation_edatas = None

    def initialize(self):
        """Initialize."""
        super().initialize()
        self.inner_solver.initialize()

    def get_inner_parameter_ids(self) -> List[str]:
        """Get the ids of the inner parameters."""
        return self.inner_problem.get_x_ids()

    def __call__(
        self,
        x_dct: Dict,
        sensi_orders: Tuple[int],
        mode: ModeType,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: List[amici.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
    ):
        """Perform the actual AMICI call, with hierarchical optimization.

        See documentation for
        `pypesto.objective.amici.amici_calculator.AmiciCalculator.__call__()`.

        The return object also includes the simulation results that were
        generated to solve the inner problem, as well as the parameters that
        solver the inner problem.
        """
        if not self.inner_problem.check_edatas(edatas=edatas):
            raise ValueError(
                'The experimental data provided to this call differs from '
                'the experimental data used to setup the hierarchical '
                'optimizer.'
            )

        if self.simulation_edatas is None:
            simulation_edatas = edatas
        else:
            simulation_edatas = self.simulation_edatas

        # ToDo: implement this check
        # else:
        #     if not self.inner_problem.check_simulation_edatas(edatas=simulation_edatas):
        #         raise ValueError(
        #             'The experimental data provided to this call for simulation differs from '
        #             'the experimental data used to setup the hierarchical '
        #             'optimizer. Only different timepoints are allowed.'
        #         )

        # compute optimal inner parameters
        x_dct = copy.deepcopy(x_dct)
        x_dct.update(self.inner_problem.get_dummy_values(scaled=True))
        inner_result = super().__call__(
            x_dct=x_dct,
            sensi_orders=(0,),
            mode=mode,
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=edatas,
            n_threads=n_threads,
            x_ids=x_ids,
            parameter_mapping=parameter_mapping,
            fim_for_hess=fim_for_hess,
        )
        inner_rdatas = inner_result[RDATAS]

        # if any amici simulation failed, it's unlikely we can compute
        # meaningful inner parameters, so we better just fail early.
        if any(rdata.status != amici.AMICI_SUCCESS for rdata in inner_rdatas):
            # if the gradient was requested, we need to provide some value
            # for it
            dim = len(x_ids)
            if 1 in sensi_orders:
                inner_result[GRAD] = np.full(shape=dim, fill_value=np.nan)
            if 2 in sensi_orders:
                inner_result[HESS] = np.full(
                    shape=(dim, dim), fill_value=np.nan
                )
            return inner_result

        inner_parameters = self.inner_solver.solve(
            problem=self.inner_problem,
            sim=[rdata[AMICI_Y] for rdata in inner_rdatas],
            sigma=[rdata[AMICI_SIGMAY] for rdata in inner_rdatas],
            scaled=True,
        )

        # fill in optimal values
        # directly writing to parameter mapping ensures that plists do not
        # include hierarchically computed parameters
        x_dct = copy.deepcopy(x_dct)
        x_dct.update(inner_parameters)

        # TODO use plist to compute only required derivatives, in
        #  `super.__call__`, `amici.parameter_mapping.fill_in_parameters`
        # TODO speed gain: if no gradient is required, then simulation can be
        #  skipped here, and rdatas can be updated in place
        #  (y, sigma, res, llh).
        result = super().__call__(
            x_dct=x_dct,
            sensi_orders=sensi_orders,
            mode=mode,
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=simulation_edatas,
            n_threads=n_threads,
            x_ids=x_ids,
            parameter_mapping=parameter_mapping,
            fim_for_hess=fim_for_hess,
        )
        result[INNER_PARAMETERS] = inner_parameters
        result[INNER_RDATAS] = inner_rdatas

        return result

    def set_simulation_edatas(
        self,
        simulation_timepoints: Sequence[Sequence[Union[float, int]]] = None,
    ):
        """Set the experimental data used for simulation.

        This is required to simulate model trajectories at more timepoints than the
        measurement timepoints. Checks ensure that the experimental data used for
        simulation is a subset of the experimental data used for the inner problem.

        Parameters
        ----------
        simulation_timepoints:
            The outer sequence should contain a sequence of timepoints for each
            experimental condition.
        """

        if simulation_timepoints is None:
            self.simulation_edatas = None
            return
        else:
            simulation_edatas = copy.deepcopy(self.inner_problem.edatas)
            for i, edata in enumerate(simulation_edatas):
                edata.setTimepoints(simulation_timepoints[i])

        self.simulation_edatas = simulation_edatas
