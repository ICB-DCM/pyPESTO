import copy
import warnings

import numpy as np

from ...C import (
    DUMMY_INNER_VALUE,
    LOWER_BOUND,
    UPPER_BOUND,
    InnerParameterType,
)
from ..base_parameter import InnerParameter


def get_finite_quotient(
    numerator: float,
    denominator: float,
    inner_parameter_type: InnerParameterType,
):
    """Get a finite value for the inner parameter.

    Parameters
    ----------
    numerator:
        The numerator of a quotient.
    denominator:
        The denominator of a quotient.
    useless:
        The value to return if the quotient is not finite.

    Returns
    -------
    `num / den` if it's finite, else a dummy value.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            quotient = float(numerator / denominator)
        if not np.isfinite(quotient):
            raise ValueError
        return quotient
    except Exception:
        return DUMMY_INNER_VALUE[inner_parameter_type]


def compute_optimal_scaling(
    data: list[np.ndarray],
    sim: list[np.ndarray],
    sigma: list[np.ndarray],
    mask: list[np.ndarray],
    optimal_offset: float = None,
) -> float:
    """
    Compute optimal scaling.

    Compute optimal scaling parameter for the given measurements and model
    outputs. See https://doi.org/10.1093/bioinformatics/btz581 SI Section 3.1
    for the derivation.
    """
    # SI Section 3.1: page 9, bottom (s = ...) with offset b=0 (as there is
    # no offset, or it already has been subtracted from the measurements)
    # numerator and denominator to compute the optimal scaling
    # num: \sum_i \frac{ \bar{y}_i * \tilde{h}_i } { \sigma_i^2 }
    # den:  \sum_i \frac{ \tilde{h}_i^2 }{ \sigma_i^2 }

    # numerator, denominator
    num, den = 0.0, 0.0

    # iterate over conditions
    for sim_i, data_i, sigma_i, mask_i in zip(sim, data, sigma, mask):
        # extract relevant values
        sim_x = sim_i[mask_i]  # \tilde{h}_i
        data_x = (
            data_i[mask_i] - optimal_offset
            if optimal_offset is not None
            else data_i[mask_i]
        )  # \bar{y}_i
        sigma_x = sigma_i[mask_i]  # \sigma_i
        # update statistics
        num += np.nansum(sim_x * data_x / sigma_x**2)
        den += np.nansum(sim_x**2 / sigma_x**2)

    return get_finite_quotient(
        numerator=num,
        denominator=den,
        inner_parameter_type=InnerParameterType.SCALING,
    )


def apply_scaling(
    scaling_value: float, sim: list[np.ndarray], mask: list[np.ndarray]
):
    """Apply scaling to simulations (in-place).

    Parameters
    ----------
    scaling_value:
        The optimal scaling for the masked simulations.
    sim:
        All full (unmasked) simulations.
    mask:
        The masks that indicate the simulation subset that corresponds to the
        `scaling_value`.
    """
    for i in range(len(sim)):
        sim[i][mask[i]] *= scaling_value


def apply_scaling_to_sensitivities(
    scaling_value: float, ssim: list[np.ndarray], mask: list[np.ndarray]
):
    """Apply scaling to sensitivities (in-place).

    Parameters
    ----------
    scaling_value:
        The optimal scaling for the masked simulations.
    ssim:
        All full (unmasked) sensitivities.
    mask:
        The masks that indicate the simulation subset that corresponds to the
        `scaling_value`.
    """
    for i in range(len(ssim)):
        ssim[i][:, mask[i]] *= scaling_value


def compute_optimal_offset(
    data: list[np.ndarray],
    sim: list[np.ndarray],
    sigma: list[np.ndarray],
    mask: list[np.ndarray],
    optimal_scaling: float = None,
) -> float:
    """Compute optimal offset.

    Compute optimal offset for the given measurements and model outputs. See
    https://doi.org/10.1093/bioinformatics/btz581 SI Section 3.1 for the
    derivation. This function handles offsets that occur without any coupled
    scaling parameter.
    """
    # SI Section 3.1: page 9, bottom (b = ...) with scaling s=1
    # numerator and denominator to compute the optimal offset
    # num: \sum_i \frac{ \bar{y}_i - \tilde{h}_i } { \sigma_i^2 }
    # den:  \sum_i \frac{1}{ \sigma_i^2 }
    num, den = 0.0, 0.0

    # iterate over conditions
    for sim_i, data_i, sigma_i, mask_i in zip(sim, data, sigma, mask):
        # extract relevant values
        sim_x = (
            optimal_scaling * sim_i[mask_i]
            if optimal_scaling is not None
            else sim_i[mask_i]
        )  # \tilde{h}_i
        data_x = data_i[mask_i]  # \bar{y}_i
        sigma_x = sigma_i[mask_i]  # \sigma_i
        # update statistics
        num += np.nansum((data_x - sim_x) / sigma_x**2)
        den += np.nansum(1 / sigma_x**2)
    return get_finite_quotient(
        numerator=num,
        denominator=den,
        inner_parameter_type=InnerParameterType.OFFSET,
    )


def compute_optimal_offset_coupled(
    data: list[np.ndarray],
    sim: list[np.ndarray],
    sigma: list[np.ndarray],
    mask: list[np.ndarray],
) -> float:
    """Compute optimal offset.

    Compute optimal offset for an observable that has both an offset and
    scaling inner parameter.

    See https://doi.org/10.1093/bioinformatics/btz581 SI Section 3.1 for the
    derivation.
    """
    # will be \sum_i \frac{ \tilde{h}_i }{ \sigma_i^2 }
    h = 0.0
    # will be \sum_i \frac{1}{ \sigma_i^2 }
    recnoise = 0.0
    # will be \sum_i \frac{ \bar{y}_i * \tilde{h}_i }{ \sigma_i^2 }
    yh = 0.0
    # will be \sum_i \frac{ \bar{y}_i }{ \sigma_i^2 }
    y = 0.0
    # will be \sum_i \frac{ \tilde{h}_i^2}{ \sigma_i^2 }
    h2 = 0.0

    # iterate over conditions
    for sim_i, data_i, sigma_i, mask_i in zip(sim, data, sigma, mask):
        if mask_i.max(initial=False) is False:
            continue
        # extract relevant values
        sim_x = sim_i[mask_i]  # \tilde{h}_i
        data_x = data_i[mask_i]  # \bar{y}_i
        sigma_x = sigma_i[mask_i]  # \sigma_i
        # update statistics
        s2 = sigma_x**2  # \sigma_i^2
        h += np.nansum(sim_x / s2)
        recnoise += np.nansum(1 / s2)
        yh += np.nansum((sim_x * data_x) / s2)
        y += np.nansum(data_x / s2)
        h2 += np.nansum((sim_x**2) / s2)

    # numerator and denominator in equation 11, each multiplied by
    #  recnoise = \sum_i \frac{1}{ \sigma_i^2 }
    num = y - (yh * h) / h2
    den = recnoise - (h**2) / h2

    # If simulation is essentially constant, then offset and scaling
    # have the same effect. In this case, we set the offset to a dummy value,
    # and estimate scaling. This is because offset is computed first, by the
    # calculator.
    # NB: this will cause issues if e.g. a simulation with dynamics has values
    #     on the order of the zero check here. An attempt to avoid this
    #     is done here by checking that the numerator is not close to zero.
    if np.isclose(den, 0, atol=1e-14) and not np.isclose(num, 0, atol=1e-4):
        # `get_finite_quotient` will now return the dummy value
        den = 0

    return get_finite_quotient(
        numerator=num,
        denominator=den,
        inner_parameter_type=InnerParameterType.OFFSET,
    )


def apply_offset(
    offset_value: float,
    data: list[np.ndarray],
    mask: list[np.ndarray],
    is_data: bool = True,
):
    """Apply offset to data (in-place).

    Acts on data instead of simulation because of:
    `data = scaling * simulation + offset`. The first step of transforming
    data and simulation so they can be compared is to either multiply
    `simulation` by `scaling`, or subtract `offset` from `data`.
    As we only currently have `compute_optimal_offset_coupled`, not
    `compute_optimal_scaling_coupled`, we need to compute optimal offset first,
    so the first step must act on data.

    NB: as this applies to data in-place, it's important to supply a copy of
    data s.t. the original data is not affected.

    Parameters
    ----------
    offset_value:
        The optimal offset for the masked simulations.
    sim:
        All full (unmasked) data.
    mask:
        The masks that indicate the data subset that corresponds to the
        `offset_value`.
    is_data:
        Whether the data is being offset, or the simulation is. If False, the
        offset is added to the simulation instead of subtracted from the data.
    """
    for i in range(len(data)):
        data[i][mask[i]] += -offset_value if is_data else offset_value


def compute_optimal_sigma(
    data: list[np.ndarray], sim: list[np.ndarray], mask: list[np.ndarray]
) -> float:
    """Compute optimal sigma.

    Compute optimal sigmas for the given measurements and model outputs. See
    https://doi.org/10.1093/bioinformatics/btz581 SI Section 3.2 for the
    derivation.
    """
    # numerator, denominator
    num, den = 0.0, 0.0

    # iterate over conditions
    for sim_i, data_i, mask_i in zip(sim, data, mask):
        # extract relevant values
        sim_x = sim_i[mask_i]
        data_x = data_i[mask_i]
        # update statistics
        num += np.nansum((data_x - sim_x) ** 2)
        den += sim_x.size

    if num == 0:
        raise AssertionError()
    # compute optimal value
    return np.sqrt(num / den)


def apply_sigma(
    sigma_value: float, sigma: list[np.ndarray], mask: list[np.ndarray]
):
    """Apply optimal sigma to pre-existing sigma arrays (in-place).

    Parameters
    ----------
    sigma_value:
        The optimal sigma value.
    sigma:
        All full (unmasked) sigmas.
    mask:
        The masks that indicate the sigma subset that corresponds to the
        `sigma_value`.
    """
    for i in range(len(sigma)):
        sigma[i][mask[i]] = sigma_value


def compute_bounded_optimal_scaling_offset_coupled(
    data: list[np.ndarray],
    sim: list[np.ndarray],
    sigma: list[np.ndarray],
    s: InnerParameter,
    b: InnerParameter,
    s_opt_value: float,
    b_opt_value: float,
):
    """Compute optimal scaling and offset of a constrained optimization problem.

    Computes the optimal scaling and offset of a constrained optimization in
    case the unconstrained optimization yields a value outside the bounds.
    We know the optimal solution then lies on the boundary of the bounds.
    In the 2D offset-scaling bounded (rectangular) space, after unconstrained
    optimization, if only one parameter is outside the bounds, then there is
    one active edge (constraint) of the rectangle. We perform optimization on
    this edge that is unconstrained in the other parameter. If this new optimum
    is outside the bounds of the other parameter, the nearest vertex is chosen
    as the optimum. If both parameters are outside the bounds, then there are
    two active edges, which are optimized independently as above, then compared.

    Parameters
    ----------
    data:
        The data.
    sim:
        The simulation.
    sigma:
        The noise parameters.
    s:
        The scaling parameter.
    b:
        The offset parameter.
    s_opt_value:
        The optimal scaling value of the unconstrained problem.
    b_opt_value:
        The optimal offset value of the unconstrained problem.

    Returns
    -------
    The optimal scaling and offset of the constrained problem.
    """
    # Define relevant data and sim
    # Make all non-masked data and sim nan's in the original one
    relevant_data = copy.deepcopy(data)
    relevant_sim = copy.deepcopy(sim)
    for i in range(len(data)):
        relevant_data[i][~s.ixs[i]] = np.nan
        relevant_sim[i][~s.ixs[i]] = np.nan

    # Get relevant data mask
    relevant_data_mask = []
    for i in range(len(data)):
        relevant_data_mask.append(~np.isnan(relevant_data[i]))

    # Get bounds
    s_bounds = s.get_bounds()
    b_bounds = b.get_bounds()

    # Get unsatisfied bounds
    s_unsatisfied = s.get_unsatisfied_bound(s_opt_value)
    b_unsatisfied = b.get_unsatisfied_bound(b_opt_value)

    # If both parameters are unsatisfied, we need to check 2
    # unconstrained problems, clip the solutions to the bounds, and
    # choose the one with the lowest objective value
    if s_unsatisfied is not None and b_unsatisfied is not None:
        # Solve the two unconstrained problems
        candidate_points = [
            (
                s_bounds[s_unsatisfied],
                np.clip(
                    compute_optimal_offset(
                        data, sim, sigma, s.ixs, s_bounds[s_unsatisfied]
                    ),
                    b_bounds[LOWER_BOUND],
                    b_bounds[UPPER_BOUND],
                ),
            ),
            (
                np.clip(
                    compute_optimal_scaling(
                        data, sim, sigma, s.ixs, b_bounds[b_unsatisfied]
                    ),
                    s_bounds[LOWER_BOUND],
                    s_bounds[UPPER_BOUND],
                ),
                b_bounds[b_unsatisfied],
            ),
        ]

        # Evaluate the objective function at the candidate points
        candidate_objective_values = [
            compute_nllh(
                data=relevant_data,
                sim=[
                    sim_i * candidate_point[0] + candidate_point[1]
                    for sim_i in relevant_sim
                ],
                sigma=sigma,
                data_mask=relevant_data_mask,
            )
            for candidate_point in candidate_points
        ]
        # The constrained solution is the candidate point with the lowest
        # objective value
        constrained_solution = candidate_points[
            np.argmin(candidate_objective_values)
        ]

    # If only one parameter is unsatisfied, we need to solve a
    # unconstrained problem, clipped to its boundary
    elif s_unsatisfied is not None:
        # Solve the unconstrained problem
        constrained_solution = (
            s_bounds[s_unsatisfied],
            np.clip(
                compute_optimal_offset(
                    data, sim, sigma, s.ixs, s_bounds[s_unsatisfied]
                ),
                b_bounds[LOWER_BOUND],
                b_bounds[UPPER_BOUND],
            ),
        )
    elif b_unsatisfied is not None:
        # Solve the unconstrained problem
        constrained_solution = (
            np.clip(
                compute_optimal_scaling(
                    data, sim, sigma, s.ixs, b_bounds[b_unsatisfied]
                ),
                s_bounds[LOWER_BOUND],
                s_bounds[UPPER_BOUND],
            ),
            b_bounds[b_unsatisfied],
        )

    return constrained_solution


def compute_nllh(
    data: list[np.ndarray],
    sim: list[np.ndarray],
    sigma: list[np.ndarray],
    data_mask: list[np.ndarray],
) -> float:
    """Compute negative log-likelihood.

    Compute negative log-likelihood of the data, given the model outputs and
    sigmas.
    """
    nllh = 0.0
    for data_i, sim_i, sigma_i, data_mask_i in zip(
        data, sim, sigma, data_mask
    ):
        # Mask the data, sim and sigma
        data_i = data_i[data_mask_i]
        sim_i = sim_i[data_mask_i]
        sigma_i = sigma_i[data_mask_i]

        # Compute the negative log-likelihood
        nllh += 0.5 * np.nansum(
            np.log(2 * np.pi * sigma_i**2)
        ) + 0.5 * np.nansum((data_i - sim_i) ** 2 / sigma_i**2)

    return nllh


def compute_nllh_gradient_for_condition(
    data: np.ndarray,
    sim: np.ndarray,
    sigma: np.ndarray,
    ssim: np.ndarray,
    ssigma: np.ndarray,
):
    """Compute gradient of negative the log-likelihood function for a condition.

    Compute gradient of negative log-likelihood function with respect to
    outer optimization parameters for a condition, given the model outputs,
    relevant data and sigmas.
    """
    return np.nansum(
        np.multiply(
            ssigma,
            ((np.full(data.shape, 1) - (data - sim) ** 2 / sigma**2) / sigma),
        ),
        axis=(1, 2),
    ) + np.nansum(
        np.multiply(ssim, (sim - data) / sigma**2),
        axis=(1, 2),
    )
