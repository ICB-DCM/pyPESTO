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

try:
    import amici
    import amici.sim.sundials as asd
except ImportError:
    amici = None

if TYPE_CHECKING:
    try:
        from amici.sim._parameter_mapping import ParameterMapping
    except ImportError:
        ParameterMapping = None

AmiciModel = Union["asd.Model", "asd.ModelPtr"]
AmiciSolver = Union["asd.Solver", "asd.SolverPtr"]


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
        edatas: list[asd.ExpData],
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
        from amici.sim.sundials.petab.v1 import fill_in_parameters

        # set order in solver
        sensi_order = 0
        if sensi_orders:
            sensi_order = max(sensi_orders)

        if sensi_order == 2 and fim_for_hess:
            # we use the FIM
            amici_solver.set_sensitivity_order(sensi_order - 1)
        else:
            amici_solver.set_sensitivity_order(sensi_order)

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
        if (
            not self._known_least_squares_safe
            and mode == MODE_RES
            and 1 in sensi_orders
        ):
            if not amici_model.get_add_sigma_residuals() and any(
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
                    "amici_model.set_add_sigma_residuals()."
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


class AmiciCalculatorPetabV2(AmiciCalculator):
    """Class to perform the AMICI call and obtain objective function values."""

    def __init__(
        self,
        petab_simulator: amici.sim.sundials.petab.PetabSimulator,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.petab_simulator = petab_simulator
        # fixed-parameter values (set by update_from_problem); merged into the
        #  simulation so the simulator doesn't fall back to PEtab nominal values
        self.fixed_parameters: dict[str, float] = {}

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
        amici_solver = self.petab_simulator._solver

        # request the full return data (llh, sllh, FIM, residuals, sres); this
        #  is the simulator default, set explicitly to be robust to prior state
        amici_solver.set_return_data_reporting_mode(asd.RDataReporting.full)

        # set the sensitivity order in the solver
        sensi_order = max(sensi_orders) if sensi_orders else 0
        if sensi_order == 2 and fim_for_hess:
            # use the FIM instead of the Hessian
            amici_solver.set_sensitivity_order(sensi_order - 1)
        else:
            amici_solver.set_sensitivity_order(sensi_order)

        dim = len(x_ids)

        # add back fixed-parameter values (stripped from x_dct by the objective)
        #  so the simulator doesn't fall back to the PEtab nominal values
        problem_parameters = {**self.fixed_parameters, **x_dct}
        result = self.petab_simulator.simulate(problem_parameters)
        rdatas = result.rdatas

        # check whether the simulation failed
        if any(rdata["status"] < 0.0 for rdata in rdatas):
            return get_error_output(
                amici_model, edatas, rdatas, sensi_orders, mode, dim
            )

        nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
            sensi_orders, mode, dim
        )
        nllh = -result.llh

        # column indices of the free optimization parameters within the
        #  estimated-parameter ordering used by result.sllh/s2llh/sres
        estimated_ids = list(self.petab_simulator._petab_problem.x_free_ids)
        free_ids = [x_id for x_id in x_ids if x_id in x_dct]

        def _free_block_indices() -> list[int]:
            try:
                return [estimated_ids.index(x_id) for x_id in free_ids]
            except ValueError as e:
                raise ValueError(
                    "Missing sensitivities for free parameters "
                    f"{set(free_ids) - set(estimated_ids)}. A free parameter "
                    "is likely constant in the AMICI model "
                    "(non_estimated_parameters_as_constants=True)."
                ) from e

        if mode == MODE_FUN:
            if 1 in sensi_orders:
                if result.sllh is None and np.isnan(result.llh):
                    snllh = np.full(len(x_ids), np.nan)
                else:
                    try:
                        # llh-gradient (dict) -> nllh-gradient (array over the
                        #  free parameters)
                        snllh = -np.array(
                            [result.sllh[x_id] for x_id in free_ids]
                        )
                    except KeyError as e:
                        raise ValueError(
                            "Cannot compute gradient, missing entry for "
                            f"{set(free_ids) - set(result.sllh)}."
                        ) from e
            if 2 in sensi_orders:
                if result.s2llh is None and np.isnan(result.llh):
                    s2nllh = np.full((len(x_ids), len(x_ids)), np.nan)
                else:
                    # result.s2llh is the FIM (~Hessian of the negative
                    #  log-likelihood, so no sign flip), ordered by the
                    #  estimated PEtab parameters; select the free block
                    sel = _free_block_indices()
                    s2nllh = result.s2llh[np.ix_(sel, sel)]
        elif mode == MODE_RES:
            if 1 in sensi_orders and not self._known_least_squares_safe:
                # least-squares residuals only match the likelihood for
                #  parameter-independent sigma, unless sigma residuals are
                #  added to the model
                if (
                    not self.petab_simulator._model.get_add_sigma_residuals()
                    and any(
                        (r["ssigmay"] is not None and np.any(r["ssigmay"]))
                        or (r["ssigmaz"] is not None and np.any(r["ssigmaz"]))
                        for r in rdatas
                    )
                ):
                    raise RuntimeError(
                        "Cannot use the least-squares solver with parameter-"
                        "dependent sigma. Enable sigma residuals on the model "
                        "via `set_add_sigma_residuals(True)`."
                    )
                self._known_least_squares_safe = True
            if 0 in sensi_orders:
                res = result.res
            if 1 in sensi_orders:
                if result.sres is None:
                    raise ValueError(
                        "Residual sensitivities (sres) were not computed."
                    )
                # result.sres has one column per estimated PEtab parameter;
                #  select the free-parameter columns
                sres = result.sres[:, _free_block_indices()]

        ret = {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas,
        }

        return filter_return_dict(ret)


def calculate_function_values(
    rdatas,
    sensi_orders: tuple[int, ...],
    mode: ModeType,
    amici_model: AmiciModel,
    amici_solver: AmiciSolver,
    edatas: list[asd.ExpData],
    x_ids: Sequence[str],
    parameter_mapping: ParameterMapping,
    fim_for_hess: bool,
):
    """Calculate the function values from rdatas and return as dict."""

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

    par_sim_ids = list(amici_model.get_free_parameter_ids())
    sensi_method = amici_solver.get_sensitivity_method()

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
                    sensi_method != asd.SensitivityMethod.forward
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
