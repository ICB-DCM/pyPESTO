from typing import List

import numpy as np

from ..C import DUMMY_INNER_VALUE, InnerParameterType


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
        quotient = float(numerator / denominator)
        if not np.isfinite(quotient):
            raise ValueError
        return quotient
    except Exception:
        return DUMMY_INNER_VALUE[InnerParameterType]


def compute_optimal_scaling(
    data: List[np.ndarray],
    sim: List[np.ndarray],
    sigma: List[np.ndarray],
    mask: List[np.ndarray],
) -> float:
    """
    Compute optimal scaling.

    Compute optimal scaling parameter for the given measurements and model
    outputs. See https://doi.org/10.1093/bioinformatics/btz581 SI Section 3.1
    for the derivation.
    """
    # numerator, denominator
    num, den = 0.0, 0.0

    # iterate over conditions
    for sim_i, data_i, sigma_i, mask_i in zip(sim, data, sigma, mask):
        # extract relevant values
        sim_x = sim_i[mask_i]
        data_x = data_i[mask_i]
        sigma_x = sigma_i[mask_i]
        # update statistics
        num += np.nansum(sim_x * data_x / sigma_x**2)
        den += np.nansum(sim_x**2 / sigma_x**2)

    return get_finite_quotient(
        numerator=num,
        denominator=den,
        inner_parameter_type=InnerParameterType.SCALING,
    )


def apply_scaling(
    scaling_value: float, sim: List[np.ndarray], mask: List[np.ndarray]
):
    """Apply scaling to simulations (in-place)."""
    for i in range(len(sim)):
        sim[i][mask[i]] = scaling_value * sim[i][mask[i]]


def compute_optimal_offset(
    data: List[np.ndarray],
    sim: List[np.ndarray],
    sigma: List[np.ndarray],
    mask: List[np.ndarray],
) -> float:
    """Compute optimal offset.

    Compute optimal offset for the given measurements and model outputs. See
    https://doi.org/10.1093/bioinformatics/btz581 SI Section 3.1 for the
    derivation.
    """
    # numerator, denominator
    num, den = 0.0, 0.0

    # iterate over conditions
    for sim_i, data_i, sigma_i, mask_i in zip(sim, data, sigma, mask):
        # extract relevant values
        sim_x = sim_i[mask_i]
        data_x = data_i[mask_i]
        sigma_x = sigma_i[mask_i]
        # update statistics
        num += np.nansum((data_x - sim_x) / sigma_x**2)
        den += np.nansum(1 / sigma_x**2)

    return get_finite_quotient(
        numerator=num,
        denominator=den,
        inner_parameter_type=InnerParameterType.OFFSET,
    )


def compute_optimal_offset_coupled(
    data: List[np.ndarray],
    sim: List[np.ndarray],
    sigma: List[np.ndarray],
    mask: List[np.ndarray],
) -> float:
    """Compute optimal offset."""
    # numerator, denominator
    h, recnoise, yh, y, h2 = 0.0, 0.0, 0.0, 0.0, 0.0

    # iterate over conditions
    for sim_i, data_i, sigma_i, mask_i in zip(sim, data, sigma, mask):
        if mask_i.max(initial=False) is False:
            continue
        # extract relevant values
        sim_x = sim_i[mask_i]
        data_x = data_i[mask_i]
        sigma_x = sigma_i[mask_i]
        # update statistics
        s2 = sigma_x**2
        h += np.nansum(sim_x / s2)
        recnoise += np.nansum(1 / s2)
        yh += np.nansum((sim_x * data_x) / s2)
        y += np.nansum(data_x / s2)
        h2 += np.nansum((sim_x**2) / s2)

    r1 = (yh * h) / h2
    r2 = (h**2) / h2
    num = y - r1
    den = recnoise - r2

    # compute optimal value
    return float(num / den)


def apply_offset(
    offset_value: float, data: List[np.ndarray], mask: List[np.ndarray]
):
    """Apply offset to simulations (in-place)."""
    for i in range(len(data)):
        data[i][mask[i]] = data[i][mask[i]] - offset_value


def compute_optimal_sigma(
    data: List[np.ndarray], sim: List[np.ndarray], mask: List[np.ndarray]
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

    # compute optimal value
    return np.sqrt(num / den)


def apply_sigma(
    sigma_value: float, sigma: List[np.ndarray], mask: List[np.ndarray]
):
    """Apply scaling to simulations (in-place)."""
    for i in range(len(sigma)):
        sigma[i][mask[i]] = sigma_value * sigma[i][mask[i]]


def compute_nllh(
    data: List[np.ndarray], sim: List[np.ndarray], sigma: List[np.ndarray]
) -> float:
    """Compute negative log-likelihood.

    Compute negative log-likelihood of the data, given the model outputs and
    sigmas.
    """
    nllh = 0.0
    for data_i, sim_i, sigma_i in zip(data, sim, sigma):
        nllh += 0.5 * np.nansum(np.log(2 * np.pi * sigma_i**2))
        nllh += 0.5 * np.nansum((data_i - sim_i) ** 2 / sigma_i**2)
    return nllh
