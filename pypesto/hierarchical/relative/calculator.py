from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np

try:
    import amici
    from amici.petab.conditions import fill_in_parameters
    from amici.petab.parameter_mapping import ParameterMapping
except ImportError:
    pass

from ...C import (
    AMICI_SIGMAY,
    AMICI_SSIGMAY,
    AMICI_SY,
    AMICI_Y,
    FVAL,
    GRAD,
    HESS,
    INNER_PARAMETERS,
    RDATAS,
    RES,
    SRES,
    ModeType,
)
from ...objective.amici.amici_calculator import (
    AmiciCalculator,
    AmiciModel,
    AmiciSolver,
)
from ...objective.amici.amici_util import (
    filter_return_dict,
    init_return_values,
)
from .problem import AmiciInnerProblem
from .solver import AnalyticalInnerSolver, InnerSolver


class RelativeAmiciCalculator(AmiciCalculator):
    """A calculator that is passed as `calculator` to the pypesto.AmiciObjective."""

    def __init__(
        self,
        inner_problem: AmiciInnerProblem,
        inner_solver: InnerSolver | None = None,
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

    def initialize(self):
        """Initialize."""
        super().initialize()
        self.inner_solver.initialize()

    def __call__(
        self,
        x_dct: dict,
        sensi_orders: tuple[int],
        mode: ModeType,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: list[amici.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
        rdatas: list[amici.ReturnData] = None,
    ):
        """Perform the actual AMICI call, with hierarchical optimization.

        The return object also includes the simulation results that were
        generated to solve the inner problem, as well as the parameters that
        solver the inner problem.

        Parameters
        ----------
        x_dct:
            Parameters for which to compute function value and derivatives.
        sensi_orders:
            Tuple of requested sensitivity orders.
        mode:
            Call mode (function value or residual based).
        amici_model:
            The AMICI model.
        amici_solver:
            The AMICI solver.
        edatas:
            The experimental data.
        n_threads:
            Number of threads for AMICI call.
        x_ids:
            Ids of optimization parameters.
        parameter_mapping:
            Mapping of optimization to simulation parameters.
        fim_for_hess:
            Whether to use the FIM (if available) instead of the Hessian (if
            requested).
        rdatas:
            AMICI simulation return data. In case the calculator is part of
            the :class:`pypesto.objective.amici.InnerCalculatorCollector`,
            it will already simulate the model and pass the results here.

        Returns
        -------
        inner_result:
            A dict containing the calculation results: FVAL, GRAD, RDATAS and INNER_PARAMETERS.
        """
        if not self.inner_problem.check_edatas(edatas=edatas):
            raise ValueError(
                "The experimental data provided to this call differs from "
                "the experimental data used to setup the hierarchical "
                "optimizer."
            )

        if (
            1 in sensi_orders
            and amici_solver.getSensitivityMethod()
            == amici.SensitivityMethod.adjoint
        ) or 2 in sensi_orders:
            inner_result, inner_parameters = self.call_amici_twice(
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
        else:
            inner_result, inner_parameters = self.calculate_directly(
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
                rdatas=rdatas,
            )

        inner_result[INNER_PARAMETERS] = (
            np.array(
                [
                    inner_parameters[x_id]
                    for x_id in self.inner_problem.get_x_ids()
                ]
            )
            if inner_parameters is not None
            else None
        )

        return inner_result

    def call_amici_twice(
        self,
        x_dct: dict,
        sensi_orders: tuple[int],
        mode: ModeType,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: list[amici.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
    ):
        """Calculate by calling AMICI twice.

        This is necessary if the adjoint method is used, or if the Hessian is
        requested. In these cases, AMICI is called first to obtain simulations
        for the calculation of the inner parameters, and then again to obtain
        the requested objective function and gradient through AMICI.
        """
        dim = len(x_ids)
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
        rdatas = inner_result[RDATAS]

        # if any amici simulation failed, it's unlikely we can compute
        # meaningful inner parameters, so we better just fail early.
        if any(rdata.status != amici.AMICI_SUCCESS for rdata in rdatas):
            # if the gradient was requested, we need to provide some value
            # for it
            if 1 in sensi_orders:
                inner_result[GRAD] = np.full(shape=dim, fill_value=np.nan)
            if 2 in sensi_orders:
                inner_result[HESS] = np.full(
                    shape=(dim, dim), fill_value=np.nan
                )
            return inner_result, None

        inner_parameters = self.inner_solver.solve(
            problem=self.inner_problem,
            sim=[rdata[AMICI_Y] for rdata in rdatas],
            sigma=[rdata[AMICI_SIGMAY] for rdata in rdatas],
            scaled=True,
        )

        # fill in optimal values
        # directly writing to parameter mapping ensures that plists do not
        # include hierarchically computed parameters
        x_dct = copy.deepcopy(x_dct)
        x_dct.update(inner_parameters)

        # TODO use plist to compute only required derivatives, in
        #  `super.__call__`, `amici.parameter_mapping.fill_in_parameters`
        inner_result = super().__call__(
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
        return inner_result, inner_parameters

    def calculate_directly(
        self,
        x_dct: dict,
        sensi_orders: tuple[int],
        mode: ModeType,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: list[amici.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
        rdatas: list[amici.ReturnData] = None,
    ):
        """Calculate directly via solver calculate methods.

        This is possible if the forward method is used, and the Hessian is not
        requested. In this case, the objective function and gradient are computed
        directly using the solver methods.
        """
        dim = len(x_ids)
        # compute optimal inner parameters
        x_dct = copy.deepcopy(x_dct)

        # initialize return values
        nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
            sensi_orders, mode, dim
        )

        # set order in solver
        sensi_order = 0
        if sensi_orders:
            sensi_order = max(sensi_orders)

        # if AMICI ReturnData is not provided, we need to simulate the model
        if rdatas is None:
            amici_solver.setSensitivityOrder(sensi_order)
            x_dct.update(self.inner_problem.get_dummy_values(scaled=True))
            # fill in parameters
            fill_in_parameters(
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

        inner_result = {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas,
        }

        # if any amici simulation failed, it's unlikely we can compute
        # meaningful inner parameters, so we better just fail early.
        if any(rdata.status != amici.AMICI_SUCCESS for rdata in rdatas):
            inner_result[FVAL] = np.inf
            if 1 in sensi_orders:
                inner_result[GRAD] = np.full(
                    shape=len(x_ids), fill_value=np.nan
                )
            return filter_return_dict(inner_result), None

        inner_parameters = self.inner_solver.solve(
            problem=self.inner_problem,
            sim=[rdata[AMICI_Y] for rdata in rdatas],
            sigma=[rdata[AMICI_SIGMAY] for rdata in rdatas],
            scaled=True,
        )

        # compute the objective function value
        inner_result[FVAL] = self.inner_solver.calculate_obj_function(
            problem=self.inner_problem,
            sim=[rdata[AMICI_Y] for rdata in rdatas],
            sigma=[rdata[AMICI_SIGMAY] for rdata in rdatas],
            inner_parameters=inner_parameters,
        )

        # compute the objective function gradient, if requested
        if 1 in sensi_orders:
            inner_result[GRAD] = self.inner_solver.calculate_gradients(
                problem=self.inner_problem,
                sim=[rdata[AMICI_Y] for rdata in rdatas],
                ssim=[rdata[AMICI_SY] for rdata in rdatas],
                sigma=[rdata[AMICI_SIGMAY] for rdata in rdatas],
                ssigma=[rdata[AMICI_SSIGMAY] for rdata in rdatas],
                inner_parameters=inner_parameters,
                parameter_mapping=parameter_mapping,
                par_opt_ids=x_ids,
                par_sim_ids=amici_model.getParameterIds(),
                snllh=snllh,
            )
        # apply the computed inner parameters to the ReturnData
        rdatas = self.inner_solver.apply_inner_parameters_to_rdatas(
            problem=self.inner_problem,
            rdatas=rdatas,
            inner_parameters=inner_parameters,
        )
        inner_result[RDATAS] = rdatas

        return inner_result, inner_parameters
