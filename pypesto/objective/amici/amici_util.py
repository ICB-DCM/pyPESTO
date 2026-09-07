from __future__ import annotations

import logging
import numbers
import warnings
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
from ...logging import log_level_active

if TYPE_CHECKING:
    try:
        import amici
        from amici.sim._parameter_mapping import (
            ParameterMapping,
            ParameterMappingForCondition,
        )
        from petab import v2
    except ImportError:
        ParameterMapping = ParameterMappingForCondition = v2 = None

AmiciModel = Union["amici.Model", "amici.ModelPtr"]
AmiciSolver = Union["amici.Solver", "amici.SolverPtr"]

logger = logging.getLogger(__name__)


def map_par_opt_to_par_sim(
    condition_map_sim_var: dict[str, float | str],
    x_dct: dict[str, float],
    amici_model: AmiciModel,
) -> np.ndarray:
    """
    Create simulation vector from optimization vector using the mapping.

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
    par_sim_vals = [
        condition_map_sim_var[par_id]
        for par_id in amici_model.get_free_parameter_ids()
    ]

    # iterate over simulation parameter indices
    for ix, val in enumerate(par_sim_vals):
        if not isinstance(val, numbers.Number):
            try:
                # value is optimization parameter id
                par_sim_vals[ix] = x_dct[val]
            except KeyError:
                # this may happen in case of states with NaN in the conditions
                #  table
                par_sim_vals[ix] = np.nan

    # return the created simulation parameter vector
    return np.array(par_sim_vals)


def create_plist_from_par_opt_to_par_sim(mapping_par_opt_to_par_sim):
    """
    Create list of parameter indices for which sensitivity is to be computed.

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
    warnings.warn(
        "This function will be removed in future releases. ",
        DeprecationWarning,
        stacklevel=2,
    )
    plist = []

    # iterate over simulation parameter indices
    for j_par_sim, val in enumerate(mapping_par_opt_to_par_sim):
        if not isinstance(val, numbers.Number):
            plist.append(j_par_sim)

    # return the created simulation parameter vector
    return plist


def create_identity_parameter_mapping(
    amici_model: AmiciModel, n_conditions: int
) -> ParameterMapping:
    """Create a dummy identity parameter mapping table.

    This fills in only the dynamic parameters. Values for fixed parameters,
    both in preequilibration and simulation, are assumed to be provided
    correctly in model or edatas already.
    """
    from amici.sim._parameter_mapping import (
        ParameterMapping,
        ParameterMappingForCondition,
    )
    from amici.sim.sundials.petab.v1._parameter_scaling import (
        amici_to_petab_scale,
    )

    x_ids = list(amici_model.get_free_parameter_ids())
    x_scales = list(amici_model.get_parameter_scale())
    parameter_mapping = ParameterMapping()
    for _ in range(n_conditions):
        condition_map_sim_var = {x_id: x_id for x_id in x_ids}
        condition_scale_map_sim_var = {
            x_id: amici_to_petab_scale(x_scale)
            for x_id, x_scale in zip(x_ids, x_scales, strict=True)
        }
        # assumes fixed parameters are filled in already
        mapping_for_condition = ParameterMappingForCondition(
            map_sim_var=condition_map_sim_var,
            scale_map_sim_var=condition_scale_map_sim_var,
        )

        parameter_mapping.append(mapping_for_condition)
    return parameter_mapping


def par_index_slices(
    par_opt_ids: Sequence[str],
    par_sim_ids: Sequence[str],
    condition_map_sim_var: dict[str, float | str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate numpy arrays for indexing based on `mapping_par_opt_to_par_sim`.

    Parameters
    ----------
    par_opt_ids:
        The optimization parameter ids. Needed for order.
    par_sim_ids:
        The simulation parameter ids. Needed for order.
    condition_map_sim_var:
        The simulation to optimization parameter mapping.

    Returns
    -------
    par_sim_slice:
        array of simulation parameter indices
    par_opt_slice:
        array of optimization parameter indices
    """
    # Create ID to index mapping for more efficient lookup than list.index
    par_opt_id_to_idx = {id_: idx for idx, id_ in enumerate(par_opt_ids)}
    par_sim_id_to_idx = {id_: idx for idx, id_ in enumerate(par_sim_ids)}

    # bool array indicating which simulation parameters map to any estimated
    #  parameters
    par_sim_maps_to_str = np.fromiter(
        (
            isinstance(condition_map_sim_var[par_id], str)
            for par_id in par_sim_ids
        ),
        dtype=bool,
        count=len(par_sim_ids),
    )
    # elsewhere, amici.ExpData.plist is set to compute only sensitivities
    #  w.r.t. estimated parameters. the parameter ordering is preserved there.
    #  therefore, the cumulative sum of mapped estimated parameters yields the
    #  index for AMICI-computed sensitivities for the respective parameter.
    cumsum_par_sim_maps_to_str = np.cumsum(par_sim_maps_to_str)

    zip_iterator = zip(
        *(
            (
                # sensitivity parameter index in AMICI simulation results
                cumsum_par_sim_maps_to_str[par_sim_id_to_idx[par_sim_id]] - 1,
                # corresponding optimization parameter index
                par_opt_id_to_idx[par_opt_id],
            )
            for par_sim_id, par_opt_id in condition_map_sim_var.items()
            if isinstance(par_opt_id, str) and par_opt_id in par_opt_id_to_idx
        ),
        strict=True,
    )
    par_sim_slice = np.fromiter(next(zip_iterator), dtype=int)
    par_opt_slice = np.fromiter(next(zip_iterator), dtype=int)
    return par_sim_slice, par_opt_slice


def petab_v2_placeholder_mapping(
    petab_problem: v2.Problem,
    experiment: v2.Experiment,
) -> dict[str, str]:
    """Map observable/noise placeholders to the parameters overriding them.

    For PEtab v2, the observable and noise placeholders are model parameters,
    overridden per measurement. Numeric overrides are omitted.

    Expects the problem as preprocessed by the AMICI PEtab importer, whose
    ``ExperimentsToSbmlConverter`` rejects overrides that differ between the
    measurements of one experiment ("timepoint-specific mappings"). Within an
    experiment a placeholder therefore has a single value, which is what makes
    this mapping well defined.
    """
    observables = {
        observable.id: observable for observable in petab_problem.observables
    }
    mapping = {}
    for measurement in petab_problem.get_measurements_for_experiment(
        experiment
    ):
        observable = observables[measurement.observable_id]
        for placeholders, overrides in (
            (
                observable.observable_placeholders,
                measurement.observable_parameters,
            ),
            (observable.noise_placeholders, measurement.noise_parameters),
        ):
            for placeholder, override in zip(
                placeholders, overrides, strict=True
            ):
                if not override.is_Symbol:
                    continue
                mapping[str(placeholder)] = str(override)
    return mapping


def petab_v2_index_slices(
    petab_problem: v2.Problem,
    par_sim_ids: Sequence[str],
    edatas: list[amici.ExpData],
    par_opt_ids: Sequence[str],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate index slices for a PEtab v2 problem, one entry per experiment.

    PEtab v2 counterpart of :func:`par_index_slices`. There is no PEtab v1
    style parameter mapping to derive the correspondence from; instead, model
    parameters carry the IDs of the PEtab problem parameters, except for the
    observable/noise placeholders, which are overridden per experiment.
    Sensitivities are computed for, and returned in the order of, the model
    parameters in ``ExpData.plist``.

    Parameters
    ----------
    petab_problem:
        The PEtab v2 problem, as preprocessed by the AMICI PEtab importer.
    par_sim_ids:
        The simulation (model) parameter ids. Needed for order.
    edatas:
        The experimental data, one per PEtab experiment, in the order of
        ``petab_problem.experiments``.
    par_opt_ids:
        The optimization parameter ids. Needed for order. Sensitivities of
        parameters that are not among them -- e.g. the parameters solved for
        in an inner problem -- are omitted from the slices.

    Returns
    -------
    One ``(par_sim_slice, par_opt_slice)`` pair per experiment, as consumed by
    :func:`add_sim_grad_to_opt_grad`'s accumulation step.
    """
    par_opt_ix = {par_id: ix for ix, par_id in enumerate(par_opt_ids)}

    index_slices = []
    for edata, experiment in zip(
        edatas, petab_problem.experiments, strict=True
    ):
        placeholder_mapping = petab_v2_placeholder_mapping(
            petab_problem, experiment
        )
        # the PEtab problem parameter behind each computed sensitivity
        problem_ids = [
            placeholder_mapping.get(par_sim_ids[ix], par_sim_ids[ix])
            for ix in edata.plist
        ]
        par_sim_slice = [
            sens_ix
            for sens_ix, par_id in enumerate(problem_ids)
            if par_id in par_opt_ix
        ]
        index_slices.append(
            (
                np.asarray(par_sim_slice, dtype=int),
                np.asarray(
                    [par_opt_ix[problem_ids[ix]] for ix in par_sim_slice],
                    dtype=int,
                ),
            )
        )
    return index_slices


def add_sim_grad_to_opt_grad(
    par_opt_ids: Sequence[str],
    par_sim_ids: Sequence[str],
    condition_map_sim_var: dict[str, float | str],
    sim_grad: np.ndarray,
    opt_grad: np.ndarray,
    coefficient: float = 1.0,
) -> None:
    """
    Sum simulation gradients to objective gradient.

    Uses the provided mapping `mapping_par_opt_to_par_sim` for summing up.

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
    par_sim_slice, par_opt_slice = par_index_slices(
        par_opt_ids, par_sim_ids, condition_map_sim_var
    )

    par_opt_slice_unique, unique_index = np.unique(
        par_opt_slice, return_index=True
    )
    opt_grad[par_opt_slice_unique] += (
        coefficient * sim_grad[par_sim_slice[unique_index]]
    )

    if par_opt_slice_unique.size < par_opt_slice.size:
        for idx in range(len(par_opt_slice)):
            if idx not in unique_index:
                opt_grad[par_opt_slice[idx]] += (
                    coefficient * sim_grad[par_sim_slice[idx]]
                )


def add_sim_hess_to_opt_hess(
    par_opt_ids: Sequence[str],
    par_sim_ids: Sequence[str],
    condition_map_sim_var: dict[str, float | str],
    sim_hess: np.ndarray,
    opt_hess: np.ndarray,
    coefficient: float = 1.0,
) -> None:
    """
    Sum simulation hessians to objective hessian.

    Parameters
    ----------
    Same as for add_sim_grad_to_opt_grad, replacing the gradients by hessians.
    """
    par_sim_slice, par_opt_slice = par_index_slices(
        par_opt_ids, par_sim_ids, condition_map_sim_var
    )

    par_opt_slice_unique, unique_index = np.unique(
        par_opt_slice, return_index=True
    )

    non_unique_indices = [
        idx for idx in range(len(par_opt_slice)) if idx not in unique_index
    ]

    opt_hess[np.ix_(par_opt_slice_unique, par_opt_slice_unique)] += (
        coefficient
        * sim_hess[
            np.ix_(par_sim_slice[unique_index], par_sim_slice[unique_index])
        ]
    )

    if par_opt_slice_unique.size < par_opt_slice.size:
        for idx in non_unique_indices:
            opt_hess[par_opt_slice[idx], par_opt_slice_unique] += (
                coefficient
                * sim_hess[par_sim_slice[idx], par_sim_slice[unique_index]]
            )
            opt_hess[par_opt_slice_unique, par_opt_slice[idx]] += (
                coefficient
                * sim_hess[par_sim_slice[unique_index], par_sim_slice[idx]]
            )
            for jdx in non_unique_indices:
                opt_hess[par_opt_slice[idx], par_opt_slice[jdx]] += (
                    coefficient
                    * sim_hess[par_sim_slice[idx], par_sim_slice[jdx]]
                )


def sim_sres_to_opt_sres(
    par_opt_ids: Sequence[str],
    par_sim_ids: Sequence[str],
    condition_map_sim_var: dict[str, float | str],
    sim_sres: np.ndarray,
    coefficient: float = 1.0,
) -> np.ndarray:
    """

    Sum simulation residual sensitivities to objective residual sensitivities.

    Parameters
    ----------
    Mostly the same as for add_sim_grad_to_opt_grad, replacing the gradients by
    residual sensitivities.
    """
    opt_sres = np.zeros((sim_sres.shape[0], len(par_opt_ids)))

    par_sim_slice, par_opt_slice = par_index_slices(
        par_opt_ids, par_sim_ids, condition_map_sim_var
    )

    par_opt_slice_unique, unique_index = np.unique(
        par_opt_slice, return_index=True
    )
    opt_sres[:, par_opt_slice_unique] += (
        coefficient * sim_sres[:, par_sim_slice[unique_index]]
    )

    if par_opt_slice_unique.size < par_opt_slice.size:
        for idx in range(len(par_opt_slice)):
            if idx not in unique_index:
                opt_sres[:, par_opt_slice[idx]] += (
                    coefficient * sim_sres[:, par_sim_slice[idx]]
                )

    return opt_sres


def log_simulation(data_ix, rdata) -> None:
    """Log the simulation results."""
    logger.debug(f"=== DATASET {data_ix} ===")
    logger.debug(f"status: {rdata['status']}")
    logger.debug(f"llh: {rdata['llh']}")

    t_steadystate = "t_steadystate"
    if t_steadystate in rdata and rdata[t_steadystate] != np.nan:
        logger.debug(f"t_steadystate: {rdata[t_steadystate]}")

    if log_level_active(logger, logging.DEBUG):
        logger.debug(f"res: {rdata['res']}")


def get_error_output(
    amici_model: AmiciModel,
    edatas: Sequence[amici.ExpData],
    rdatas: Sequence[amici.ReturnData],
    sensi_orders: tuple[int, ...],
    mode: ModeType,
    dim: int,
) -> dict:
    """Get default output upon error.

    Returns values indicative of an error, that is with nan entries in all
    vectors, and a function value, i.e. nllh, of `np.inf`.
    """
    if not amici_model.nt():
        nt = sum(data.nt() for data in edatas)
    else:
        nt = sum(
            data.nt() if data.nt() else amici_model.nt() for data in edatas
        )
    n_res = nt * amici_model.nytrue
    if amici_model.get_add_sigma_residuals():
        n_res *= 2

    nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
        sensi_orders, mode, dim, True
    )
    if res is not None:
        res = np.nan * np.ones(n_res)
    if sres is not None:
        sres = np.nan * np.ones([n_res, dim])

    ret = {
        FVAL: nllh,
        GRAD: snllh,
        HESS: s2nllh,
        RES: res,
        SRES: sres,
        RDATAS: rdatas,
    }
    return filter_return_dict(ret)


def init_return_values(
    sensi_orders: tuple[int, ...],
    mode: ModeType,
    dim: int,
    error: bool = False,
):
    """Initialize return values."""
    if error:
        fval = np.inf
        sval = np.nan
    else:
        fval = sval = 0.0

    nllh = fval
    snllh = None
    s2nllh = None
    if mode == MODE_FUN:
        if 1 in sensi_orders:
            snllh = sval * np.ones(dim)
        if 2 in sensi_orders:
            s2nllh = sval * np.ones([dim, dim])

    chi2 = None
    res = None
    sres = None
    if mode == MODE_RES:
        if 0 in sensi_orders:
            chi2 = fval
            res = np.zeros([0])
        if 1 in sensi_orders:
            sres = np.zeros([0, dim])

    return nllh, snllh, s2nllh, chi2, res, sres


def filter_return_dict(ret) -> dict:
    """Filter return dict for non-None values."""
    return {key: val for key, val in ret.items() if val is not None}
