from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

import numpy as np

from ...C import (
    AMICI_CHI2,
    AMICI_FIM,
    AMICI_LLH,
    AMICI_RES,
    AMICI_SCHI2,
    AMICI_SLLH,
    AMICI_SRES,
    AMICI_SSIGMAY,
    AMICI_SSIGMAZ,
    FVAL,
    GRAD,
    HESS,
    MODE_FUN,
    MODE_RES,
    RDATAS,
    RES,
    SRES,
    ModeType,
)
from .amici_util import (
    add_sim_grad_to_opt_grad,
    add_sim_hess_to_opt_hess,
    filter_return_dict,
    get_error_output,
    init_return_values,
    log_simulation,
    sim_sres_to_opt_sres,
)

if TYPE_CHECKING:
    try:
        import amici
        from amici.petab.parameter_mapping import ParameterMapping
    except ImportError:
        ParameterMapping = None

AmiciModel = Union["amici.Model", "amici.ModelPtr"]
AmiciSolver = Union["amici.Solver", "amici.SolverPtr"]


class AmiciCalculator:
    """Class to perform the AMICI call and obtain objective function values."""

    def __init__(self):
        self._known_least_squares_safe = False

    def initialize(self):
        """Initialize the calculator. Default: Do nothing."""

    def check_least_squares_safe(
        self, amici_model, rdatas, sensi_orders, mode, mse_for_fval
    ):
        """Check if the least squares solver is safe to use."""
        if self._known_least_squares_safe:
            return  # already checked

        if 1 not in sensi_orders or not (mode == MODE_RES or mse_for_fval):
            return  # no need to check

        if not amici_model.getAddSigmaResiduals() and any(
            (
                (r[AMICI_SSIGMAY] is not None and np.any(r[AMICI_SSIGMAY]))
                or (r[AMICI_SSIGMAZ] is not None and np.any(r[AMICI_SSIGMAZ]))
            )
            for r in rdatas
        ):
            if mode == MODE_RES:
                raise RuntimeError(
                    "Cannot use least squares solver with parameter "
                    "dependent sigma! Support can be enabled via "
                    "amici_model.setAddSigmaResiduals()."
                )
            if mse_for_fval:
                raise RuntimeError(
                    "Cannot use mean squared error for function value "
                    "with parameter dependent sigma! Support can be "
                    "enabled via amici_model.setAddSigmaResiduals()"
                )

        self._known_least_squares_safe = True  # don't check this again

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
        mse_for_fval: bool,
    ):
        """Perform the actual AMICI call.

        Called within the :func:`AmiciObjective.__call__` method.

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
        mse_for_fval:
            Whether to use (negative!) mean squared error for the function value.
        """
        import amici.petab.conditions

        # set order in solver
        sensi_order = 0
        if sensi_orders:
            sensi_order = max(sensi_orders)

        if sensi_order == 2 and fim_for_hess:
            # we use the FIM
            amici_solver.setSensitivityOrder(sensi_order - 1)
        else:
            amici_solver.setSensitivityOrder(sensi_order)

        # fill in parameters
        amici.petab.conditions.fill_in_parameters(
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
        self.check_least_squares_safe(
            amici_model, rdatas, sensi_orders, mode, mse_for_fval
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
            mse_for_fval=mse_for_fval,
        )


def calculate_function_values(
    rdatas,
    sensi_orders: tuple[int, ...],
    mode: ModeType,
    amici_model: AmiciModel,
    amici_solver: AmiciSolver,
    edatas: list[amici.ExpData],
    x_ids: Sequence[str],
    parameter_mapping: ParameterMapping,
    fim_for_hess: bool,
    mse_for_fval: bool,
):
    """Calculate the function values from rdatas and return as dict."""
    import amici

    # full optimization problem dimension (including fixed parameters)
    dim = len(x_ids)

    # check if the simulation failed
    if any(rdata["status"] < 0.0 for rdata in rdatas):
        return get_error_output(
            amici_model, edatas, rdatas, sensi_orders, mode, dim
        )

    fval, grad, hess, res, sres = init_return_values(sensi_orders, mode, dim)

    par_sim_ids = list(amici_model.getParameterIds())
    sensi_method = amici_solver.getSensitivityMethod()

    ndata = sum(
        np.sum(np.isfinite(edata.getObservedData())) for edata in edatas
    )
    # iterate over return data
    for data_ix, rdata in enumerate(rdatas):
        log_simulation(data_ix, rdata)

        condition_map_sim_var = parameter_mapping[data_ix].map_sim_var

        # add objective value]

        fval_field = AMICI_CHI2 if mse_for_fval else AMICI_LLH
        fval -= rdata[fval_field]

        if mode == MODE_FUN:
            if not np.isfinite(fval):
                return get_error_output(
                    amici_model, edatas, rdatas, sensi_orders, mode, dim
                )

            if 1 in sensi_orders:
                # add gradient

                grad_field = AMICI_SCHI2 if mse_for_fval else AMICI_SLLH
                add_sim_grad_to_opt_grad(
                    x_ids,
                    par_sim_ids,
                    condition_map_sim_var,
                    rdata[grad_field],
                    grad,
                    coefficient=-1.0,
                )

                if not np.isfinite(grad).all():
                    return get_error_output(
                        amici_model, edatas, rdatas, sensi_orders, mode, dim
                    )

                # Hessian
            if 2 in sensi_orders:
                if (
                    sensi_method != amici.SensitivityMethod_forward
                    or not fim_for_hess
                ):
                    raise ValueError("AMICI cannot compute Hessians yet.")
                    # add FIM for Hessian
                add_sim_hess_to_opt_hess(
                    x_ids,
                    par_sim_ids,
                    condition_map_sim_var,
                    rdata[AMICI_FIM],
                    hess,
                    coefficient=+1.0,
                )
                if not np.isfinite(hess).all():
                    return get_error_output(
                        amici_model, edatas, rdatas, sensi_orders, mode, dim
                    )

        elif mode == MODE_RES:
            if 0 in sensi_orders:
                res = (
                    np.hstack([res, rdata[AMICI_RES]])
                    if res.size
                    else rdata[AMICI_RES]
                )
            if 1 in sensi_orders:
                opt_sres = sim_sres_to_opt_sres(
                    x_ids,
                    par_sim_ids,
                    condition_map_sim_var,
                    rdata[AMICI_SRES],
                    coefficient=1.0,
                )
                sres = np.vstack([sres, opt_sres]) if sres.size else opt_sres

    if mse_for_fval:
        # normalize with number of data points to convert SSE to MSE
        fval /= ndata
        grad /= ndata
        hess /= ndata

    ret = {
        FVAL: fval,
        GRAD: grad,
        HESS: hess,
        RES: res,
        SRES: sres,
        RDATAS: rdatas,
    }

    return filter_return_dict(ret)
