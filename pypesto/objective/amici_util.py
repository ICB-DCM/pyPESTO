import numpy as np
import numbers
from typing import Dict, Sequence, Union
import logging

from .constants import FVAL, GRAD, HESS, RES, SRES, RDATAS

try:
    import amici
    import amici.petab_objective
    import amici.parameter_mapping
    from amici.parameter_mapping import (
        ParameterMapping, ParameterMappingForCondition)
except ImportError:
    pass

AmiciModel = Union['amici.Model', 'amici.ModelPtr']
AmiciSolver = Union['amici.Solver', 'amici.SolverPtr']

logger = logging.getLogger(__name__)


def map_par_opt_to_par_sim(
        condition_map_sim_var: Dict[str, Union[float, str]],
        x_dct: Dict[str, float],
        amici_model: AmiciModel
) -> np.ndarray:
    """
    From the optimization vector, create the simulation vector according
    to the mapping.

    Parameters
    ----------
    condition_map_sim_var:
        Simulation to optimization parameter mapping.
    x_dct:
        The optimization parameters dict.
    amici_model:
        The amici model.

    Returns
    -------
    par_sim_vals:
        The simulation parameters vector corresponding to x under the
        specified mapping.
    """
    par_sim_vals = [condition_map_sim_var[par_id]
                    for par_id in amici_model.getParameterIds()]

    # iterate over simulation parameter indices
    for ix, val in enumerate(par_sim_vals):
        if not isinstance(val, numbers.Number):
            # value is optimization parameter id
            par_sim_vals[ix] = x_dct[val]

    # return the created simulation parameter vector
    return np.array(par_sim_vals)


def create_plist_from_par_opt_to_par_sim(mapping_par_opt_to_par_sim):
    """
    From the parameter mapping `mapping_par_opt_to_par_sim`, create the
    simulation plist according to the mapping `mapping`.

    Parameters
    ----------

    mapping_par_opt_to_par_sim: array-like of str
        len == n_par_sim, the entries are either numeric, or
        optimization parameter ids.

    Returns
    -------
    plist: array-like of float
        List of parameter indices for which the sensitivity needs to be
        computed
    """
    plist = []

    # iterate over simulation parameter indices
    for j_par_sim, val in enumerate(mapping_par_opt_to_par_sim):
        if not isinstance(val, numbers.Number):
            plist.append(j_par_sim)

    # return the created simulation parameter vector
    return plist


def create_identity_parameter_mapping(
        amici_model: AmiciModel, n_conditions: int
) -> 'ParameterMapping':
    """Create a dummy identity parameter mapping table.

    This fills in only the dynamic parameters. Values for fixed parameters,
    both in preequilibration and simulation, are assumed to be provided
    correctly in model or edatas already.
    """
    x_ids = list(amici_model.getParameterIds())
    x_scales = list(amici_model.getParameterScale())
    parameter_mapping = ParameterMapping()
    for _ in range(n_conditions):
        condition_map_sim_var = {x_id: x_id for x_id in x_ids}
        condition_scale_map_sim_var = {
            x_id: amici.parameter_mapping.amici_to_petab_scale(x_scale)
            for x_id, x_scale in zip(x_ids, x_scales)}
        # assumes fixed parameters are filled in already
        mapping_for_condition = ParameterMappingForCondition(
            map_sim_var=condition_map_sim_var,
            scale_map_sim_var=condition_scale_map_sim_var)

        parameter_mapping.append(mapping_for_condition)
    return parameter_mapping


def add_sim_grad_to_opt_grad(
        par_opt_ids: Sequence[str],
        par_sim_ids: Sequence[str],
        condition_map_sim_var: Dict[str, Union[float, str]],
        sim_grad: Sequence[float],
        opt_grad: Sequence[float],
        coefficient: float = 1.0):
    """
    Sum simulation gradients to objective gradient according to the provided
    mapping `mapping_par_opt_to_par_sim`.

    Parameters
    ----------
    par_opt_ids:
        The optimization parameter ids. Needed for order.
    par_sim_ids:
        The simulation parameter ids. Needed for order.
    condition_map_sim_var:
        The simulation to optimization parameter mapping.
    sim_grad:
        Simulation gradient.
    opt_grad:
        The optimization gradient. To which sim_grad is added.
        Changed in-place.
    coefficient:
        Coefficient for sim_grad when adding to opt_grad.
    """
    for par_sim, par_opt in condition_map_sim_var.items():
        if not isinstance(par_opt, str):
            continue
        par_sim_idx = par_sim_ids.index(par_sim)
        par_opt_idx = par_opt_ids.index(par_opt)

        opt_grad[par_opt_idx] += coefficient * sim_grad[par_sim_idx]


def add_sim_hess_to_opt_hess(
        par_opt_ids: Sequence[str],
        par_sim_ids: Sequence[str],
        condition_map_sim_var: Dict[str, Union[float, str]],
        sim_hess: np.ndarray,
        opt_hess: np.ndarray,
        coefficient: float = 1.0):
    """
    Sum simulation hessians to objective hessian according to the provided
    mapping `mapping_par_opt_to_par_sim`.

    Parameters
    ----------
    Same as for add_sim_grad_to_opt_grad, replacing the gradients by hessians.
    """
    for par_sim_id, par_opt_id in condition_map_sim_var.items():
        if not isinstance(par_opt_id, str):
            continue
        par_sim_idx = par_sim_ids.index(par_sim_id)
        par_opt_idx = par_opt_ids.index(par_opt_id)

        for par_sim_id_2, par_opt_id_2 in condition_map_sim_var.items():
            if not isinstance(par_opt_id_2, str):
                continue
            par_sim_idx_2 = par_sim_ids.index(par_sim_id_2)
            par_opt_idx_2 = par_opt_ids.index(par_opt_id_2)

            opt_hess[par_opt_idx, par_opt_idx_2] += \
                coefficient * sim_hess[par_sim_idx, par_sim_idx_2]


def sim_sres_to_opt_sres(par_opt_ids: Sequence[str],
                         par_sim_ids: Sequence[str],
                         condition_map_sim_var: Dict[str, Union[float, str]],
                         sim_sres: np.ndarray,
                         coefficient: float = 1.0):
    """
    Sum simulation residual sensitivities to objective residual sensitivities
    according to the provided mapping.

    Parameters
    ----------
    Mostly the same as for add_sim_grad_to_opt_grad, replacing the gradients by
    residual sensitivities.
    """
    opt_sres = np.zeros((sim_sres.shape[0], len(par_opt_ids)))

    for par_sim_id, par_opt_id in condition_map_sim_var.items():
        if not isinstance(par_opt_id, str):
            continue

        par_sim_idx = par_sim_ids.index(par_sim_id)
        par_opt_idx = par_opt_ids.index(par_opt_id)
        opt_sres[:, par_opt_idx] += \
            coefficient * sim_sres[:, par_sim_idx]

    return opt_sres


def log_simulation(data_ix, rdata):
    """Log the simulation results."""
    logger.debug(f"=== DATASET {data_ix} ===")
    logger.debug(f"status: {rdata['status']}")
    logger.debug(f"llh: {rdata['llh']}")

    t_steadystate = 't_steadystate'
    if t_steadystate in rdata and rdata[t_steadystate] != np.nan:
        logger.debug(f"t_steadystate: {rdata[t_steadystate]}")

    logger.debug(f"res: {rdata['res']}")


def get_error_output(
        amici_model: AmiciModel,
        edatas: Sequence['amici.ExpData'],
        rdatas: Sequence['amici.ReturnData'],
        dim: int):
    """Default output upon error.

    Returns values indicative of an error, that is with nan entries in all
    vectors, and a function value, i.e. nllh, of `np.inf`.
    """
    if not amici_model.nt():
        nt = sum(data.nt() for data in edatas)
    else:
        nt = sum(data.nt() if data.nt() else amici_model.nt()
                 for data in edatas)
    n_res = nt * amici_model.nytrue

    return {
        FVAL: np.inf,
        GRAD: np.nan * np.ones(dim),
        HESS: np.nan * np.ones([dim, dim]),
        RES:  np.nan * np.ones(n_res),
        SRES: np.nan * np.ones([n_res, dim]),
        RDATAS: rdatas
    }
