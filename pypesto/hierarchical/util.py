import numpy as np
import copy
from typing import Dict, List


def compute_optimal_scaling(
        data: List[np.ndarray],
        sim: List[np.ndarray],
        sigma: List[np.ndarray],
        mask: List[np.ndarray]) -> float:
    """Compute optimal scaling."""
    # numerator, denominator
    num, den = 0.0, 0.0

    # iterate over conditions
    for sim_i, data_i, sigma_i, mask_i in \
            zip(sim, data, sigma, mask):
        # extract relevant values
        sim_x = sim_i[mask_i]
        data_x = data_i[mask_i]
        sigma_x = sigma_i[mask_i]
        # update statistics
        num += np.nansum(sim_x * data_x / sigma_x ** 2)
        den += np.nansum(sim_x ** 2 / sigma_x ** 2)

    # compute optimal value
    x_opt = 1.0  # value doesn't matter
    if not np.isclose(den, 0.0):
        x_opt = num / den

    return float(x_opt)


def apply_scaling(
        scaling_value: float,
        sim: List[np.ndarray],
        mask: List[np.ndarray]):
    """Apply scaling to simulations (in-place)."""
    for i in range(len(sim)):
        sim[i][mask[i]] = scaling_value * sim[i][mask[i]]


def compute_optimal_offset(
        data: List[np.ndarray],
        sim: List[np.ndarray],
        sigma: List[np.ndarray],
        mask: List[np.ndarray]) -> float:
    """Compute optimal scaling."""
    # numerator, denominator
    num, den = 0.0, 0.0

    # iterate over conditions
    for sim_i, data_i, sigma_i, mask_i in \
            zip(sim, data, sigma, mask):
        # extract relevant values
        sim_x = sim_i[mask_i]
        data_x = data_i[mask_i]
        sigma_x = sigma_i[mask_i]
        # update statistics
        num += np.nansum((data_x - sim_x) / sigma_x ** 2)
        den += np.nansum(1 / sigma_x ** 2)

    # compute optimal value
    x_opt = 0.0  # value doesn't matter
    if not np.isclose(den, 0.0):
        x_opt = num / den

    return float(x_opt)


def apply_offset(
        offset_value: float,
        sim: List[np.ndarray],
        mask: List[np.ndarray]):
    """Apply offset to simulations (in-place)."""
    for i in range(len(sim)):
        sim[i][mask[i]] = sim[i][mask[i]] + offset_value


def compute_optimal_sigma(
        data: List[np.ndarray],
        sim: List[np.ndarray],
        mask: List[np.ndarray]) -> float:
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
    x_opt = 1.0  # value doesn't matter
    if not np.isclose(x_opt, 0.0):
        # we report the standard deviation, not the variance
        x_opt = np.sqrt(num / den)

    return float(x_opt)


def apply_sigma(
        sigma_value: float,
        sigma: List[np.ndarray],
        mask: List[np.ndarray]):
    """Apply scaling to simulations (in-place)."""
    for i in range(len(sigma)):
        sigma[i][mask[i]] = sigma_value * sigma[i][mask[i]]


def compute_nllh(
        data: List[np.ndarray],
        sim: List[np.ndarray],
        sigma: List[np.ndarray]) -> float:
    nllh = 0.0
    for data_i, sim_i, sigma_i in zip(data, sim, sigma):
        nllh += 0.5 * np.nansum(np.log(2*np.pi*sigma_i**2))
        nllh += 0.5 * np.nansum((data_i-sim_i)**2 / sigma_i**2)
    return nllh