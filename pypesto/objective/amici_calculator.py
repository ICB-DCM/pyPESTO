import numpy as np
from typing import Dict, List, Sequence, Union

from .constants import (
    MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, RDATAS, CHI2
)
from .amici_util import (
    add_sim_grad_to_opt_grad, add_sim_hess_to_opt_hess,
    sim_sres_to_opt_sres, log_simulation, get_error_output, filter_return_dict,
    init_return_values
)

try:
    import amici
    import amici.petab_objective
    import amici.parameter_mapping
    from amici.parameter_mapping import ParameterMapping
except ImportError:
    pass

AmiciModel = Union['amici.Model', 'amici.ModelPtr']
AmiciSolver = Union['amici.Solver', 'amici.SolverPtr']


class AmiciCalculator:
    """
    Class to perform the actual call to AMICI and obtain requested objective
    function values.
    """

    def __init__(self):
        self._known_least_squares_safe = False

    def initialize(self):
        """Initialize the calculator. Default: Do nothing."""

    def __call__(self,
                 x_dct: Dict,
                 sensi_order: int,
                 mode: str,
                 amici_model: AmiciModel,
                 amici_solver: AmiciSolver,
                 edatas: List['amici.ExpData'],
                 n_threads: int,
                 x_ids: Sequence[str],
                 parameter_mapping: 'ParameterMapping',
                 fim_for_hess: bool,
                 chunk_size: int = None):
        """Perform the actual AMICI call.

        Called within the :func:`AmiciObjective.__call__` method.

        Parameters
        ----------
        x_dct:
            Parameters for which to compute function value and derivatives.
        sensi_order:
            Maximum sensitivity order.
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
        chunk_size:
            Size of chunks the ExpData is sliced into.
        """
        # set order in solver
        if sensi_order == 2 and fim_for_hess:
            # we use the FIM
            amici_solver.setSensitivityOrder(sensi_order-1)
        else:
            amici_solver.setSensitivityOrder(sensi_order)

        # fill in parameters
        # TODO (#226) use plist to compute only required derivatives
        amici.parameter_mapping.fill_in_parameters(
            edatas=edatas,
            problem_parameters=x_dct,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=amici_model
        )
        rdatas = get_output(edatas=edatas,
                            amici_model=amici_model,
                            amici_solver=amici_solver,
                            chunk_size=chunk_size,
                            mode=mode,
                            n_threads=n_threads)

        if not self._known_least_squares_safe and mode == MODE_RES and \
                sensi_order > 0:
            if any(
                ((r['ssigmay'] is not None and np.any(r['ssigmay']))
                 or
                 (r['ssigmaz'] is not None and np.any(r['ssigmaz'])))
                for r in rdatas
            ):
                raise RuntimeError('Cannot use least squares solver with'
                                   'parameter dependent sigma!')
            self._known_least_squares_safe = True  # don't check this again

        return calculate_function_values(
            rdatas=rdatas, sensi_order=sensi_order, mode=mode,
            amici_model=amici_model, amici_solver=amici_solver, edatas=edatas,
            x_ids=x_ids, parameter_mapping=parameter_mapping,
            fim_for_hess=fim_for_hess)


def calculate_function_values(rdatas,
                              sensi_order: int,
                              mode: str,
                              amici_model: AmiciModel,
                              amici_solver: AmiciSolver,
                              edatas: List['amici.ExpData'],
                              x_ids: Sequence[str],
                              parameter_mapping: 'ParameterMapping',
                              fim_for_hess: bool):
    # full optimization problem dimension (including fixed parameters)
    dim = len(x_ids)

    # check if the simulation failed
    if any(rdata['status'] < 0.0 for rdata in rdatas):
        return get_error_output(amici_model, edatas, rdatas,
                                sensi_order, mode, dim)

    nllh, snllh, s2nllh, chi2, res, sres = init_return_values(sensi_order,
                                                              mode, dim)

    par_sim_ids = list(amici_model.getParameterIds())
    sensi_method = amici_solver.getSensitivityMethod()

    # iterate over return data
    for data_ix, rdata in enumerate(rdatas):
        log_simulation(data_ix, rdata)

        condition_map_sim_var = \
            parameter_mapping[data_ix].map_sim_var

        # add objective value
        nllh -= rdata['llh']

        if mode == MODE_FUN:
            if not np.isfinite(nllh):
                return get_error_output(amici_model, edatas, rdatas,
                                        sensi_order, mode, dim)

            if sensi_order > 0:
                # add gradient
                add_sim_grad_to_opt_grad(
                    x_ids,
                    par_sim_ids,
                    condition_map_sim_var,
                    rdata['sllh'],
                    snllh,
                    coefficient=-1.0
                )

                if not np.isfinite(snllh).all():
                    return get_error_output(amici_model, edatas, rdatas,
                                            sensi_order, mode, dim)

                # Hessian
                if sensi_order > 1:
                    if sensi_method == amici.SensitivityMethod_forward \
                            and fim_for_hess:
                        # add FIM for Hessian
                        add_sim_hess_to_opt_hess(
                            x_ids,
                            par_sim_ids,
                            condition_map_sim_var,
                            rdata['FIM'],
                            s2nllh,
                            coefficient=+1.0
                        )
                        if not np.isfinite(s2nllh).all():
                            return get_error_output(amici_model, edatas,
                                                    rdatas, sensi_order,
                                                    mode, dim)
                    else:
                        raise ValueError("AMICI cannot compute Hessians yet.")

        elif mode == MODE_RES:
            chi2 += rdata['chi2']
            res = np.hstack([res, rdata['res']]) \
                if res.size else rdata['res']
            if sensi_order > 0:
                opt_sres = sim_sres_to_opt_sres(
                    x_ids,
                    par_sim_ids,
                    condition_map_sim_var,
                    rdata['sres'],
                    coefficient=1.0
                )
                sres = np.vstack([sres, opt_sres]) \
                    if sres.size else opt_sres

    ret = {
        FVAL: nllh,
        CHI2: chi2,
        GRAD: snllh,
        HESS: s2nllh,
        RES: res,
        SRES: sres,
        RDATAS: rdatas
    }
    return filter_return_dict(ret)

def get_output(edatas: List['amici.ExpData'],
               amici_model: AmiciModel,
               amici_solver: AmiciSolver,
               chunk_size: int,
               mode: str,
               n_threads: int):
    rdatas = []
    # spliting the ExpData in chunks and simulating chunk wise
    # Do we have a maximum number of simulations allowed?
    n_edatas = len(edatas)
    if chunk_size is None:
        # simulate all conditions at once
        n_simulations = 1
    else:
        # simulate only a subset of conditions
        n_simulations = int(np.ceil(len(edatas) / chunk_size))
    # get keys based on mode
    if mode == MODE_FUN:
        requested_keys = ['llh', 'sllh', 'status']
    elif mode == MODE_RES:
        requested_keys = ['res', 'sres', 'status']
    else:
        requested_keys = None

    for i_sim in range(n_simulations):
        # slice out the conditions we actually want
        if chunk_size is None:
            ids = slice(0, n_edatas)
        else:
            ids = slice(i_sim * chunk_size,
                        min((i_sim + 1) * chunk_size, n_edatas))

        chunk = amici.runAmiciSimulations(
            amici_model,
            amici_solver,
            edatas[ids],
            num_threads=min(n_threads, len(edatas)))
        if requested_keys is not None:
            rdatas.extend([{key: bit[key] for key in requested_keys} for bit in chunk])
        else:
            # run amici simulation
            rdatas.extend(amici.runAmiciSimulations(
                amici_model,
                amici_solver,
                edatas[ids],
                num_threads=min(n_threads, len(edatas)),
            ))
        return rdatas
