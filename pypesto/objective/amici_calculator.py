import numpy as np
from typing import Dict, List, Sequence, Union

from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, RDATAS
from .amici_util import (
    add_sim_grad_to_opt_grad, add_sim_hess_to_opt_hess,
    sim_sres_to_opt_sres, log_simulation, get_error_output)

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

    def __call__(self,
                 x_dct: Dict,
                 sensi_order: int,
                 mode: str,
                 amici_model: AmiciModel,
                 amici_solver: AmiciSolver,
                 edatas: List['amici.ExpData'],
                 n_threads: int,
                 x_ids: Sequence[str],
                 parameter_mapping: ParameterMapping):
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
        """
        # full optimization problem dimension (including fixed parameters)
        dim = len(x_ids)

        # prepare outputs
        nllh = 0.0
        snllh = np.zeros(dim)
        s2nllh = np.zeros([dim, dim])

        res = np.zeros([0])
        sres = np.zeros([0, dim])

        # set order in solver
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

        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            amici_model,
            amici_solver,
            edatas,
            num_threads=min(n_threads, len(edatas)),
        )

        par_sim_ids = list(amici_model.getParameterIds())
        sensi_method = amici_solver.getSensitivityMethod()

        for data_ix, rdata in enumerate(rdatas):
            log_simulation(data_ix, rdata)

            # check if the computation failed
            if rdata['status'] < 0.0:
                return get_error_output(
                    amici_model, edatas, rdatas, dim)

            condition_map_sim_var = \
                parameter_mapping[data_ix].map_sim_var

            nllh -= rdata['llh']

            # compute objective
            if mode == MODE_FUN:

                if sensi_order > 0:
                    add_sim_grad_to_opt_grad(
                        x_ids,
                        par_sim_ids,
                        condition_map_sim_var,
                        rdata['sllh'],
                        snllh,
                        coefficient=-1.0
                    )
                    if sensi_method == 1:
                        # TODO Compute the full Hessian, and check here
                        add_sim_hess_to_opt_hess(
                            x_ids,
                            par_sim_ids,
                            condition_map_sim_var,
                            rdata['FIM'],
                            s2nllh,
                            coefficient=+1.0
                        )

            elif mode == MODE_RES:
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

        return {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas
        }
