from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

import numpy as np

from ..C import FVAL, GRAD, HESS, MODE_RES, RDATAS, RES, SRES, ModeType, RDATAS, X
from ..objective.amici.amici_calculator import (
    AmiciCalculator,
    calculate_function_values,
    AmiciModel,
    AmiciSolver,
)
from ..objective.amici.amici_util import get_error_output
from .problem import InnerProblem
from .solver import AnalyticalInnerSolver, InnerSolver
from .util import compute_nllh

import amici
from amici.parameter_mapping import ParameterMapping


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

    def solve_x_inner(
        self,
        x_dct: Dict,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: List[amici.ExpData],
        n_threads: int,
        parameter_mapping: ParameterMapping,
    ):
        if not self.inner_problem.check_edatas(edatas=edatas):
            raise ValueError('The experimental data provided to this call differs from the experimental data used to setup the hierarchical optimizer.')

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

        sim = [rdata['y'] for rdata in rdatas]
        sigma = [rdata['sigmay'] for rdata in rdatas]

        # compute optimal inner parameters
        x_inner_opt = self.inner_solver.solve(
            self.inner_problem, sim, sigma, scaled=True
        )

        return {X: x_inner_opt, RDATAS: rdatas}


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
        if not self.inner_problem.check_edatas(edatas=edatas):
            raise ValueError('The experimental data provided to this call differs from the experimental data used to setup the hierarchical optimizer.')

        inner_result = self.solve_x_inner(
            x_dct=x_dct,
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=edatas,
            n_threads=n_threads,
            parameter_mapping=parameter_mapping,
        )

        if not sensi_orders or max(sensi_orders) == 0:
            nllh = compute_nllh(
                data=self.inner_problem.data,
                sim=[rdata['y'] for rdata in inner_result[RDATAS]],
                sigma=[rdata['sigmay'] for rdata in inner_result[RDATAS]],
            )
            dim = len(x_ids)
            return {
                FVAL: nllh,
                GRAD: np.zeros(dim),
                HESS: np.zeros([dim, dim]),
                RES: np.zeros([0]),
                SRES: np.zeros([0, dim]),
                RDATAS: inner_result[RDATAS],
                'inner_parameters': inner_result[X],
            }

        # fill in optimal values
        # TODO: x_inner_opt is different for hierarchical and
        #  qualitative approach. For now I commented the following
        #  lines out to make qualitative approach work.
        # directly writing to parameter mapping ensures that plists do not
        # include hierarchically computed parameters
        x_dct = copy.deepcopy(x_dct)
        for key, val in inner_result[X].items():
            x_dct[key] = val

        # TODO use plist to compute only required derivatives, in
        # `super.__call__`, `amici.parameter_mapping.fill_in_parameters`
        result = super().__call__(
            x_dct=x_dct,
            sensi_orders=sensi_orders,
            mode=mode,
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=edatas,
            n_threads=n_threads,
            x_ids=x_ids,
            parameter_mapping=parameter_mapping,
            fim_for_hess=fim_for_hess,
        )
        result['inner_parameters'] = inner_result[X]
        result['inner_rdatas'] = inner_result[RDATAS]
        return result
