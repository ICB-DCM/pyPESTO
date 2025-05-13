from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np

from ...C import (
    AMICI_SIGMAY,
    AMICI_SSIGMAY,
    AMICI_SY,
    AMICI_Y,
    FVAL,
    GRAD,
    HESS,
    INNER_PARAMETERS,
    MODE_RES,
    RDATAS,
    RES,
    SPLINE_KNOTS,
    SRES,
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
from .problem import SemiquantProblem
from .solver import SemiquantInnerSolver

try:
    import amici
    from amici.petab.conditions import fill_in_parameters
    from amici.petab.parameter_mapping import ParameterMapping
except ImportError:
    pass


class SemiquantCalculator(AmiciCalculator):
    """A calculator is passed as `calculator` to the pypesto.AmiciObjective.

    The object is called by :func:`pypesto.AmiciObjective.call_unprocessed`
    to calculate the current objective function values and gradient.
    """

    def __init__(
        self,
        inner_problem: SemiquantProblem,
        inner_solver: SemiquantInnerSolver = None,
    ):
        """Initialize the calculator from the given problem.

        Parameters
        ----------
        inner_problem:
            The optimal scaling inner problem.
        inner_solver:
            A solver to solve ``inner_problem``.
            Defaults to ``SplineInnerSolver``.
        """
        super().__init__()
        self.inner_problem = inner_problem

        if inner_solver is None:
            inner_solver = SemiquantInnerSolver()
        self.inner_solver = inner_solver

    def initialize(self):
        """Initialize."""
        self.inner_solver.initialize()
        self.inner_problem.initialize()

    def __call__(
        self,
        x_dct: dict,
        sensi_orders: tuple[int, ...],
        mode: str,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: list[amici.ExpData],
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: ParameterMapping,
        fim_for_hess: bool,
        rdatas: list[amici.ReturnData] = None,
    ):
        """Perform the actual AMICI call.

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
            A dict containing the calculation results: FVAL, GRAD, RDATAS,
            INNER_PARAMETERS, and SPLINE_KNOTS.
        """
        if mode == MODE_RES:
            raise ValueError(
                "Spline approximation cannot be called with residual mode."
            )
        if 2 in sensi_orders:
            raise ValueError(
                "Hessian and FIM are not implemented for the spline calculator."
            )

        # get dimension of outer problem
        dim = len(x_ids)

        # initialize return values
        nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
            sensi_orders, mode, dim
        )
        # set order in solver
        sensi_order = 0
        if sensi_orders:
            sensi_order = max(sensi_orders)

        # If AMICI ReturnData is not provided, we need to simulate the model

        if rdatas is None:
            amici_solver.setSensitivityOrder(sensi_order)

            x_dct = copy.deepcopy(x_dct)
            x_dct.update(
                self.inner_problem.get_noise_dummy_values(scaled=True)
            )

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
                inner_result[GRAD] = np.full(shape=dim, fill_value=np.nan)
            return filter_return_dict(inner_result)

        sim = [rdata[AMICI_Y] for rdata in rdatas]
        sigma = [rdata[AMICI_SIGMAY] for rdata in rdatas]

        # Clip negative simulation values to zero, to avoid numerical issues.
        for i in range(len(sim)):
            sim[i] = sim[i].clip(min=0)

        # compute optimal inner parameters
        x_inner_opt = self.inner_solver.solve(self.inner_problem, sim, sigma)
        inner_result[FVAL] = self.inner_solver.calculate_obj_function(
            x_inner_opt
        )
        inner_result[SPLINE_KNOTS] = self.inner_problem.get_spline_knots()

        inner_result[INNER_PARAMETERS] = (
            self.inner_problem.get_inner_noise_parameters()
        )

        # Calculate analytical gradients if requested
        if sensi_order > 0:
            sy = [rdata[AMICI_SY] for rdata in rdatas]
            ssigma = [rdata[AMICI_SSIGMAY] for rdata in rdatas]
            inner_result[GRAD] = self.inner_solver.calculate_gradients(
                problem=self.inner_problem,
                x_inner_opt=x_inner_opt,
                sim=sim,
                amici_sigma=sigma,
                sy=sy,
                amici_ssigma=ssigma,
                parameter_mapping=parameter_mapping,
                par_opt_ids=x_ids,
                par_sim_ids=amici_model.getParameterIds(),
                par_edatas_indices=[edata.plist for edata in edatas],
                snllh=snllh,
            )

        return filter_return_dict(inner_result)
