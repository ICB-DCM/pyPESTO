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
    """Perform the AMICI call for a PEtab v2 problem.

    For PEtab v2, the mapping between PEtab problem parameters and AMICI model
    parameters lives entirely inside
    :class:`amici.sim.sundials.petab.PetabSimulator`; there is no PEtab v1
    style ``ParameterMapping`` that :class:`AmiciCalculator` could use. This
    calculator therefore delegates simulation *and* the aggregation of the
    results across experiments to the simulator and only translates the
    outcome into pyPESTO's return dict.

    The simulator is expected to operate on the same model and solver
    instances as the objective (see :class:`AmiciPetabV2Objective`), so that
    the sensitivity order and the return-data reporting mode set by the
    objective take effect in the simulation.
    """

    def __init__(
        self,
        petab_simulator: amici.sim.sundials.petab.PetabSimulator,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.petab_simulator = petab_simulator
        #: IDs of the parameters that are free in the pyPESTO problem, i.e.,
        #: those for which sensitivities are required. Set by
        #: :class:`AmiciPetabV2Objective` as soon as the objective is part of
        #: a :class:`pypesto.Problem`; ``None`` disables the check for
        #: missing sensitivities.
        self.free_parameter_ids: set[str] | None = None

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
        # set order in solver
        sensi_order = 0
        if sensi_orders:
            sensi_order = max(sensi_orders)

        if sensi_order == 2 and fim_for_hess:
            # we use the FIM
            amici_solver.set_sensitivity_order(sensi_order - 1)
        else:
            amici_solver.set_sensitivity_order(sensi_order)

        # full optimization problem dimension (including fixed parameters)
        dim = len(x_ids)

        # `result.sllh`, `result.s2llh` and `result.sres` refer to the
        #  parameters estimated in the *PEtab* problem. Parameters that are
        #  fixed in the pyPESTO problem are still estimated in the PEtab
        #  problem; parameters that are not estimated in the PEtab problem have
        #  no sensitivities and keep their zero entries below (they can only be
        #  fixed in the pyPESTO problem, see the check further down).
        # TODO: sensitivities are computed for all parameters estimated in the
        #  PEtab problem, also for those fixed in the pyPESTO problem. Unlike
        #  for PEtab v1, `plist` cannot be narrowed down from here, since it is
        #  set by `ExperimentManager.apply_parameters`.
        petab_ix = {
            x_id: ix
            for ix, x_id in enumerate(
                self.petab_simulator.exp_man.petab_problem.x_free_ids
            )
        }
        # positions in the pyPESTO parameter vector and the corresponding
        #  positions in the PEtab sensitivity arrays
        opt_ix_sel = [ix for ix, x_id in enumerate(x_ids) if x_id in petab_ix]
        sim_ix_sel = [petab_ix[x_ids[ix]] for ix in opt_ix_sel]

        if self.free_parameter_ids is not None and sensi_order > 0:
            # a parameter that is free in the pyPESTO problem, but not
            #  estimated in the PEtab problem, is a constant in the AMICI model
            #  (non_estimated_parameters_as_constants=True) -- there are no
            #  sensitivities for it, and only sensi_order 0 is supported
            if missing := self.free_parameter_ids - set(petab_ix):
                raise ValueError(
                    f"Cannot compute gradient, missing entry for {missing}. "
                    "Those parameters are not estimated in the PEtab problem "
                    "-- fix them in the pyPESTO problem, or request "
                    "`sensi_orders=(0,)` only."
                )

        # run amici simulation; parameter mapping and the aggregation of the
        #  results across experiments are handled by the PEtab simulator
        result = self.petab_simulator.simulate(x_dct)
        rdatas = result.rdatas
        # the simulator creates its own ExpData objects
        edatas = result.edatas

        for data_ix, rdata in enumerate(rdatas):
            log_simulation(data_ix, rdata)

        # check if the simulation failed
        if any(rdata["status"] < 0.0 for rdata in rdatas):
            return get_error_output(
                amici_model, edatas, rdatas, sensi_orders, mode, dim
            )

        nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
            sensi_orders, mode, dim
        )
        nllh = -result.llh

        if mode == MODE_FUN and not np.isfinite(nllh):
            return get_error_output(
                amici_model, edatas, rdatas, sensi_orders, mode, dim
            )

        # `result.sllh`, `result.s2llh` and `result.sres` refer to the
        #  parameters estimated in the *PEtab* problem. Parameters that are
        #  fixed in the pyPESTO problem are still estimated in the PEtab
        #  problem; parameters that are not estimated in the PEtab problem have
        #  no sensitivities and keep their zero entries here (they are always
        #  fixed in the pyPESTO problem, and are dropped downstream).
        # TODO: sensitivities are computed for all parameters estimated in the
        #  PEtab problem, also for those fixed in the pyPESTO problem. Unlike
        #  for PEtab v1, `plist` cannot be narrowed down from here, since it is
        #  set by `ExperimentManager.apply_parameters`.
        petab_free_ids = self.petab_simulator.exp_man.petab_problem.x_free_ids
        petab_ix = {x_id: ix for ix, x_id in enumerate(petab_free_ids)}
        # positions in the pyPESTO parameter vector and the corresponding
        #  positions in the PEtab sensitivity arrays
        opt_ix_sel = [ix for ix, x_id in enumerate(x_ids) if x_id in petab_ix]
        sim_ix_sel = [petab_ix[x_ids[ix]] for ix in opt_ix_sel]

        if (
            self.free_parameter_ids is not None
            and sensi_orders
            and max(sensi_orders) > 0
        ):
            # a parameter that is free in the pyPESTO problem, but not
            #  estimated in the PEtab problem, is a constant in the AMICI model
            #  (non_estimated_parameters_as_constants=True) -- there are no
            #  sensitivities for it, and only sensi_order 0 is supported
            if missing := self.free_parameter_ids - set(petab_ix):
                raise ValueError(
                    f"Cannot compute gradient, missing entry for {missing}. "
                    "Those parameters are not estimated in the PEtab problem "
                    "-- fix them in the pyPESTO problem, or request "
                    "`sensi_orders=(0,)` only."
                )

        if mode == MODE_FUN:
            if 1 in sensi_orders:
                if missing := {x_ids[ix] for ix in opt_ix_sel} - set(
                    result.sllh or {}
                ):
                    raise ValueError(
                        f"Cannot compute gradient, missing entry for {missing}."
                    )
                # llh to nllh, dict to array
                snllh[opt_ix_sel] = [
                    -result.sllh[x_ids[ix]] for ix in opt_ix_sel
                ]
            if 2 in sensi_orders:
                if result.s2llh is None:
                    raise ValueError("The Hessian (FIM) was not computed.")
                # `result.s2llh` is the FIM, i.e. an approximation of the
                #  Hessian of the *negative* log-likelihood -- no sign flip
                s2nllh[np.ix_(opt_ix_sel, opt_ix_sel)] = result.s2llh[
                    np.ix_(sim_ix_sel, sim_ix_sel)
                ]
        elif mode == MODE_RES:
            if 1 in sensi_orders and not self._known_least_squares_safe:
                # the least-squares residuals only match the likelihood for
                #  parameter-independent sigma, unless sigma residuals are
                #  added to the model
                if not amici_model.get_add_sigma_residuals() and any(
                    (
                        (r["ssigmay"] is not None and np.any(r["ssigmay"]))
                        or (r["ssigmaz"] is not None and np.any(r["ssigmaz"]))
                    )
                    for r in rdatas
                ):
                    raise RuntimeError(
                        "Cannot use the least-squares solver with parameter-"
                        "dependent sigma. Enable sigma residuals on the model "
                        "via `set_add_sigma_residuals(True)`."
                    )
                self._known_least_squares_safe = True  # don't check this again
            if 0 in sensi_orders:
                if (res := result.res) is None:
                    raise ValueError("The residuals were not computed.")
            if 1 in sensi_orders:
                if result.sres is None:
                    raise ValueError(
                        "The residual sensitivities were not computed."
                    )
                sres = np.zeros((result.sres.shape[0], dim))
                sres[:, opt_ix_sel] = result.sres[:, sim_ix_sel]

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
