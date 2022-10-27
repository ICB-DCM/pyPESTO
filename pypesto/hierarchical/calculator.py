from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

import numpy as np

from ..C import FVAL, GRAD, HESS, MODE_RES, RDATAS, RES, SRES, ModeType, RDATAS, X, AMICI_Y, AMICI_SIGMAY, INNER_PARAMETERS, INNER_RDATAS
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

        # compute optimal inner parameters
        x_dct = copy.deepcopy(x_dct)
        x_dct.update(self.inner_problem.get_boring_pars(scaled=True))
        inner_rdatas = super().__call__(
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
        )[RDATAS]
        inner_parameters = self.inner_solver.solve(
            problem=self.inner_problem,
            sim=[rdata[AMICI_Y] for rdata in inner_rdatas],
            sigma=[rdata[AMICI_SIGMAY] for rdata in inner_rdatas],
            scaled=True,
        )

        # fill in optimal values
        # TODO: x_inner_opt is different for hierarchical and
        #  qualitative approach. For now I commented the following
        #  lines out to make qualitative approach work.
        # directly writing to parameter mapping ensures that plists do not
        # include hierarchically computed parameters
        x_dct = copy.deepcopy(x_dct)
        x_dct.update(inner_parameters)

        # TODO use plist to compute only required derivatives, in
        # `super.__call__`, `amici.parameter_mapping.fill_in_parameters`
        # TODO speed gain: if no offset or scaling parameters, only
        #      sigma parameters in the inner problem, then simulation can be
        #      skipped here, since observables will not change.
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
        result[INNER_PARAMETERS] = inner_parameters
        result[INNER_RDATAS] = inner_rdatas

        return result
