import numpy as np
import logging
from typing import Tuple
from tqdm import tqdm

from ..result import Result
from .diagnostics import geweke_test

logger = logging.getLogger(__name__)


def calculate_ci(result: Result,
                 alpha: float = 0.95
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate parameter confidence intervals based on MCMC samples.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    alpha:
        Lower tail probability, defaults to 95% interval.

    Returns
    -------
    lb, ub:
        Bounds of the MCMC percentile-based confidence interval.
    """
    # Check if burn in index is available
    if result.sample_result.burn_in is None:
        geweke_test(result)

    # Get burn in index
    burn_in = result.sample_result.burn_in

    # Get converged parameter samples as numpy arrays
    chain = np.asarray(result.sample_result.trace_x[0, burn_in:, :])

    # Get percentile values corresponding to alpha
    percentiles = 100 * np.array([(1-alpha)/2, 1-(1-alpha)/2])

    # Get samples' upper and lower bounds
    lb, ub = np.percentile(chain, percentiles, axis=0)

    return lb, ub


def evaluate_samples(result: Result,
                     stepsize: int = 1
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate/Simulate MCMC samples.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    stepsize:
        Only one in `stepsize` values is simulated for the intervals
        generation. Recommended for long MCMC chains. Defaults to 1.

    Returns
    -------
    evaluation_observables:
        Simulated model states.
    evaluation_states:
        Simulated model observables.
    """

    # Check if burn in index is available
    if result.sample_result.burn_in is None:
        geweke_test(result)

    # Get burn in index
    burn_in = result.sample_result.burn_in

    # Get full parameter samples as numpy arrays
    arr_param = np.array(result.sample_result.trace_x[0])

    # Get converged parameter samples as numpy arrays
    chain = arr_param[np.arange(burn_in, len(arr_param), stepsize)]

    # Number of samples to be simulated
    nsamples = chain.shape[0]

    # Evaluate median of MCMC samples
    _res = result.problem.objective(np.median(chain),
                                    return_dict=True)['rdatas']
    n_conditions = len(_res)

    # create variable to store simulated model states
    n_timepoints_for_states = _res[0].x.shape[0]
    n_states = _res[0].x.shape[1]
    evaluation_states = np.empty([n_conditions,
                                  nsamples,
                                  n_timepoints_for_states,
                                  n_states])

    # create variable to store simulated model observables
    n_timepoints_for_obs = _res[0].y.shape[0]
    n_obs = _res[0].y.shape[1]
    evaluation_observables = np.empty([n_conditions,
                                       nsamples,
                                       n_timepoints_for_obs,
                                       n_obs])
    # Loop over samples
    for sample in tqdm(range(nsamples)):
        # Simulate model
        simulation = result.problem.objective(chain[sample, :],
                                              return_dict=True)['rdatas']
        # Loop over experimental conditions
        for n_cond in range(n_conditions):
            # Store model states
            evaluation_states[n_cond, sample, :, :] = simulation[n_cond].x
            # Store model observables
            evaluation_observables[n_cond, sample, :, :] = simulation[n_cond].y

    return evaluation_observables, evaluation_states


def calculate_prediction_profiles(simulated_values: np.ndarray,
                                  alpha: float = 0.95
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction confidence intervals based on MCMC samples.

    Parameters
    ----------
    simulated_values:
        Simulated model states or model observables.
    alpha:
        Lower tail probability, defaults to 95% interval.

    Returns
    -------
    lb, ub:
        Bounds of the MCMC-based prediction confidence interval.
    """

    # Get percentile values corresponding to alpha
    percentiles = 100 * np.array([(1-alpha)/2, 1-(1-alpha)/2])

    # Get samples' upper and lower bounds
    lb, ub = np.percentile(simulated_values, percentiles, axis=1)

    return lb, ub
