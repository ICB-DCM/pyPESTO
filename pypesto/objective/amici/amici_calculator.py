from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

import numpy as np

from ...C import (
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
        if (
            not self._known_least_squares_safe
            and mode == MODE_RES
            and 1 in sensi_orders
        ):
            if not amici_model.getAddSigmaResiduals() and any(
                (
                    (r["ssigmay"] is not None and np.any(r["ssigmay"]))
                    or (r["ssigmaz"] is not None and np.any(r["ssigmaz"]))
                )
                for r in rdatas
            ):
                raise RuntimeError(
                    "Cannot use least squares solver with"
                    "parameter dependent sigma! Support can be "
                    "enabled via "
                    "amici_model.setAddSigmaResiduals()."
                )
            self._known_least_squares_safe = True  # don't check this again

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

    nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
        sensi_orders, mode, dim
    )

    par_sim_ids = list(amici_model.getParameterIds())
    sensi_method = amici_solver.getSensitivityMethod()

    # iterate over return data
    for data_ix, rdata in enumerate(rdatas):
        log_simulation(data_ix, rdata)

        condition_map_sim_var = parameter_mapping[data_ix].map_sim_var

        # add objective value
        nllh -= rdata["llh"]

        if mode == MODE_FUN:
            if not np.isfinite(nllh):
                return get_error_output(
                    amici_model, edatas, rdatas, sensi_orders, mode, dim
                )

            if 1 in sensi_orders:
                # add gradient
                add_sim_grad_to_opt_grad(
                    x_ids,
                    par_sim_ids,
                    condition_map_sim_var,
                    rdata["sllh"],
                    snllh,
                    coefficient=-1.0,
                )

                if not np.isfinite(snllh).all():
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
                    rdata["FIM"],
                    s2nllh,
                    coefficient=+1.0,
                )
                if not np.isfinite(s2nllh).all():
                    return get_error_output(
                        amici_model, edatas, rdatas, sensi_orders, mode, dim
                    )

        elif mode == MODE_RES:
            if 0 in sensi_orders:
                chi2 += rdata["chi2"]
                res = (
                    np.hstack([res, rdata["res"]])
                    if res.size
                    else rdata["res"]
                )
            if 1 in sensi_orders:
                opt_sres = sim_sres_to_opt_sres(
                    x_ids,
                    par_sim_ids,
                    condition_map_sim_var,
                    rdata["sres"],
                    coefficient=1.0,
                )
                sres = np.vstack([sres, opt_sres]) if sres.size else opt_sres

    ret = {
        FVAL: nllh,
        GRAD: snllh,
        HESS: s2nllh,
        RES: res,
        SRES: sres,
        RDATAS: rdatas,
    }

    return filter_return_dict(ret)
