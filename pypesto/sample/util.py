"""A set of helper functions"""
import numpy as np
import logging
import os
from tqdm import tqdm
from typing import Tuple

from ..result import Result
from .diagnostics import geweke_test

logger = logging.getLogger(__name__)

from multiprocessing import Pool
from multiprocessing import sharedctypes
global_evaluation_states = None
global_evaluation_obs = None

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


def evaluate_samples(
        result: Result,
        stepsize: int = 1,
        n_procs: int = 1,
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
    n_procs:
        The number of processors to use, to parallelize the evaluation.

    Returns
    -------
    evaluation_obs:
        Simulated model observables.
    evaluation_states:
        Simulated model states.
    """
    # Default value of `n_procs` is the number of all available CPUs.
    if not n_procs in range(1, os.cpu_count() + 1):
        n_procs = os.cpu_count()
        logger.info(
            f'n_procs is not in range(1, {os.cpu_count()+1}). '
            f'Using {n_procs} instead.'
        )

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
    n_samples = chain.shape[0]

    # Evaluate median of MCMC samples
    _res = result.problem.objective(np.median(chain),
                                    return_dict=True)['rdatas']
    n_conditions = len(_res)

    # create variable to store simulated model states
    n_timepoints_for_states = _res[0].x.shape[0]
    n_states = _res[0].x.shape[1]
    evaluation_states_shape = (
        n_conditions,
        n_samples,
        n_timepoints_for_states,
        n_states,
    )

    # create variable to store simulated model observables
    n_timepoints_for_obs = _res[0].y.shape[0]
    n_obs = _res[0].y.shape[1]
    evaluation_obs_shape = (
        n_conditions,
        n_samples,
        n_timepoints_for_obs,
        n_obs,
    )

    if n_procs == 1:
        evaluation_states = np.empty(evaluation_states_shape)
        evaluation_obs = np.empty(evaluation_obs_shape)

        # Loop over samples
        for sample in tqdm(range(n_samples)):
            # Simulate model
            simulation = result.problem.objective(chain[sample, :],
                                                  return_dict=True)['rdatas']
            # Loop over experimental conditions
            for n_cond in range(n_conditions):
                # Store model states
                evaluation_states[n_cond, sample, :, :] = \
                    simulation[n_cond].x
                # Store model observables
                evaluation_obs[n_cond, sample, :, :] = \
                    simulation[n_cond].y
    else:
        global global_evaluation_states
        global global_evaluation_obs
        evaluation_states_ctypes = np.ctypeslib.as_ctypes(np.empty(
            evaluation_states_shape, dtype=np.float64
        ))
        evaluation_obs_ctypes = np.ctypeslib.as_ctypes(np.empty(
            evaluation_obs_shape, dtype=np.float64
        ))
        global_evaluation_states = sharedctypes.RawArray(
            evaluation_states_ctypes._type_, evaluation_states_ctypes
        )
        global_evaluation_obs = sharedctypes.RawArray(
            evaluation_obs_ctypes._type_, evaluation_obs_ctypes
        )
        tasks = [
            (sample, result, chain, n_conditions,)
            for sample in range(n_samples)
        ]
        with Pool(n_procs) as pool:
            pool.starmap(update_evaluation, tasks)
        evaluation_states = np.ctypeslib.as_array(global_evaluation_states)
        evaluation_obs = np.ctypeslib.as_array(global_evaluation_obs)

    return evaluation_obs, evaluation_states


def update_evaluation(
        sample,
        result,
        chain,
        n_conditions,
):
    global global_evaluation_states
    global global_evaluation_obs

    evaluation_states = np.ctypeslib.as_array(global_evaluation_states)
    evaluation_obs = np.ctypeslib.as_array(global_evaluation_obs)

    simulation = result.problem.objective(
        chain[sample, :],
        return_dict=True
    )['rdatas']

    for n_cond in range(n_conditions):
        evaluation_states[n_cond, sample, :, :] = simulation[n_cond].x
        evaluation_obs[n_cond, sample, :, :] = simulation[n_cond].y


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
