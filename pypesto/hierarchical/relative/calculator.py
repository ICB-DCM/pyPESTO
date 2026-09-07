from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np

try:
    import amici.sim.sundials as asd
    from amici.sim._parameter_mapping import ParameterMapping
    from amici.sim.sundials.petab.v1 import fill_in_parameters
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
    LIN,
    MODE_RES,
    RDATAS,
    RES,
    SRES,
    InnerParameterType,
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
from ..base_problem import scale_back_value_dict
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
        edatas: list[asd.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
        rdatas: list[asd.ReturnData] = None,
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

        # residual mode needs AMICI's own residuals, which only the
        #  second simulation of the two-call scheme can provide:
        #  `calculate_directly` computes the objective and its gradient from
        #  the inner solver and leaves `res`/`sres` at their empty defaults.
        if (
            (
                1 in sensi_orders
                and amici_solver.get_sensitivity_method()
                == asd.SensitivityMethod.adjoint
            )
            or 2 in sensi_orders
            or mode == MODE_RES
        ):
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
        edatas: list[asd.ExpData],
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
        # Same restriction as the parameter-dependent-sigma check in
        #  `AmiciCalculator.__call__`, and blocking the same thing: the
        #  residual sensitivities a least-squares solver needs. The residuals
        #  themselves stay available. Checked here rather than in `__call__`
        #  so that the PEtab v2 collector, which calls this directly, is
        #  covered too.
        if (
            mode == MODE_RES
            and 1 in sensi_orders
            and self.inner_problem.get_xs_for_type(InnerParameterType.SIGMA)
        ):
            raise RuntimeError(
                "Cannot use least squares solver with hierarchically "
                "estimated sigma! At the analytically optimal sigma the sum "
                "of squared residuals equals the number of measurements "
                "whatever the outer parameters are, so the least-squares "
                "objective the residuals define is constant. Estimate the "
                "sigmas as ordinary parameters to optimize in residual mode."
            )

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
        if any(rdata.status != asd.AMICI_SUCCESS for rdata in rdatas):
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

        # Fill the optimal values into the parameter mapping rather than
        #  into `x_dct`. Mapping a simulation parameter to a value instead of
        #  to an id keeps it out of AMICI's plist, so the second simulation
        #  differentiates with respect to the outer parameters only. Without
        #  this, sigmas solved for hierarchically look parameter-dependent to
        #  the least-squares check in `AmiciCalculator.__call__`.
        inner_values = scale_back_value_dict(
            inner_parameters, self.inner_problem
        )
        parameter_mapping = copy.deepcopy(parameter_mapping)
        for condition_mapping in parameter_mapping:
            for sim_par, mapped in list(condition_mapping.map_sim_var.items()):
                if isinstance(mapped, str) and mapped in inner_values:
                    condition_mapping.map_sim_var[sim_par] = inner_values[
                        mapped
                    ]
                    condition_mapping.scale_map_sim_var[sim_par] = LIN
        # the inner parameters are no longer referenced by the mapping, so
        #  leaving them in `x_dct` would make them unused problem parameters
        x_dct = {
            par_id: value
            for par_id, value in x_dct.items()
            if par_id not in inner_values
        }

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
        edatas: list[asd.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
        rdatas: list[asd.ReturnData] = None,
    ):
        """Calculate directly via solver calculate methods.

        This is possible if the forward method is used, and neither the
        Hessian nor residuals are requested. In this case, the objective
        function and gradient are computed directly using the solver methods.

        Only ``FVAL`` and ``GRAD`` are filled in; ``RES`` and ``SRES`` keep
        the empty arrays from :func:`init_return_values`, so this must not be
        called in :obj:`MODE_RES`.
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
            amici_solver.set_sensitivity_order(sensi_order)
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
            rdatas = asd.run_simulations(
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
        if any(rdata.status != asd.AMICI_SUCCESS for rdata in rdatas):
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
                par_sim_ids=amici_model.get_free_parameter_ids(),
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
