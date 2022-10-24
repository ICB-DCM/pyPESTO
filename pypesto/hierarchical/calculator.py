from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

import numpy as np

from ..C import FVAL, GRAD, HESS, MODE_RES, RDATAS, RES, SRES, ModeType
from ..objective.amici.amici_calculator import (
    AmiciCalculator,
    calculate_function_values,
)
from ..objective.amici.amici_util import get_error_output
from .problem import InnerProblem
from .solver import AnalyticalInnerSolver, InnerSolver
from .util import compute_nllh

if TYPE_CHECKING:
    try:
        import amici
        from amici.parameter_mapping import ParameterMapping
    except ImportError:
        ParameterMapping = None

AmiciModel = Union['amici.Model', 'amici.ModelPtr']
AmiciSolver = Union['amici.Solver', 'amici.SolverPtr']


class HierarchicalAmiciCalculator(AmiciCalculator):
    """
    A calculator that is passed as `calculator` to the pypesto.AmiciObjective.

    While this class cannot be used directly, it has two subclasses
    which allow to use forward or adjoint sensitivity analysis to
    solve a `pypesto.HierarchicalProblem` efficiently in an inner loop,
    while the outer optimization is only concerned with variables not
    specified as `pypesto.HierarchicalParameter`s.
    """

    def __init__(
        self,
        inner_problem: InnerProblem,
        inner_solver: InnerSolver = None,
    ):
        """Initialize the calculator from the given problem."""
        super().__init__()

        self.inner_problem = inner_problem

        if inner_solver is None:
            # inner_solver = NumericalInnerSolver()
            inner_solver = AnalyticalInnerSolver()
        self.inner_solver = inner_solver

    def initialize(self):
        """Initialize."""
        super().initialize()
        self.inner_solver.initialize()

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
        import amici.parameter_mapping

        # set order in solver
        if sensi_orders:
            sensi_order = max(sensi_orders)
        else:
            sensi_order = 0

        # if sensi_order == 2 and fim_for_hess:
        #    # we use the FIM
        #    amici_solver.setSensitivityOrder(sensi_order - 1)
        # else:
        #    amici_solver.setSensitivityOrder(sensi_order)
        dim = len(x_ids)

        # set order in solver to 0
        amici_solver.setSensitivityOrder(0)

        # fill in boring values
        x_dct = copy.deepcopy(x_dct)
        for key, val in self.inner_problem.get_boring_pars(
            scaled=True
        ).items():
            x_dct[key] = val

        # fill in parameters
        amici.parameter_mapping.fill_in_parameters(
            edatas=edatas,
            problem_parameters=x_dct,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=amici_model,
        )

        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            amici_model,
            amici_solver,
            edatas,
            num_threads=min(n_threads, len(edatas)),
        )
        if (
            not self._known_least_squares_safe
            and mode == MODE_RES
            and 1 in sensi_orders
        ):
            if not amici_model.getAddSigmaResiduals() and any(
                (
                    (r['ssigmay'] is not None and np.any(r['ssigmay']))
                    or (r['ssigmaz'] is not None and np.any(r['ssigmaz']))
                )
                for r in rdatas
            ):
                raise RuntimeError(
                    'Cannot use least squares solver with'
                    'parameter dependent sigma! Support can be '
                    'enabled via '
                    'amici_model.setAddSigmaResiduals().'
                )
            self._known_least_squares_safe = True  # don't check this again

        # check if any simulation failed
        if any(rdata['status'] < 0.0 for rdata in rdatas):
            return get_error_output(
                amici_model, edatas, rdatas, sensi_orders, mode, dim
            )

        sim = [rdata['y'] for rdata in rdatas]
        sigma = [rdata['sigmay'] for rdata in rdatas]

        # compute optimal inner parameters
        x_inner_opt = self.inner_solver.solve(
            self.inner_problem, sim, sigma, scaled=True
        )

        # nllh = self.inner_solver.calculate_obj_function(x_inner_opt)

        # print(x_inner_opt)

        # if sensi_order == 0:
        #     dim = len(x_ids)
        #     nllh = compute_nllh(self.inner_problem.data, sim, sigma)
        #     return {
        #         FVAL: nllh,
        #         GRAD: np.zeros(dim),
        #         HESS: np.zeros([dim, dim]),
        #         RES: np.zeros([0]),
        #         SRES: np.zeros([0, dim]),
        #         RDATAS: rdatas
        #     }

        # fill in optimal values
        # TODO: x_inner_opt is different for hierarchical and
        #  qualitative approach. For now I commented the following
        #  lines out to make qualitative approach work.

        # directly writing to parameter mapping ensures that plists do not
        # include hierarchically computed parameters
        # x_dct = copy.deepcopy(x_dct)
        for key, val in x_inner_opt.items():
            x_dct[key] = val

        # fill in parameters
        # TODO use plist to compute only required derivatives
        amici.parameter_mapping.fill_in_parameters(
            edatas=edatas,
            problem_parameters=x_dct,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=amici_model,
        )

        if sensi_order == 0:
            nllh = compute_nllh(self.inner_problem.data, sim, sigma)
            return {
                FVAL: nllh,
                GRAD: np.zeros(dim),
                HESS: np.zeros([dim, dim]),
                RES: np.zeros([0]),
                SRES: np.zeros([0, dim]),
                RDATAS: rdatas,
                'inner_parameters': x_inner_opt,
            }

        amici_solver.setSensitivityOrder(sensi_order)

        # resimulate
        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            amici_model,
            amici_solver,
            edatas,
            num_threads=min(n_threads, len(edatas)),
        )

        return calculate_function_values(
            rdatas=rdatas,
            sensi_orders=sensi_orders,
            mode=mode,
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=edatas,
            x_ids=x_ids,
            parameter_mapping=parameter_mapping,
            fim_for_hess=fim_for_hess,
        )
