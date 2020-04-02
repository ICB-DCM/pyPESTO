from typing import Callable, Tuple, Sequence, Dict
import numpy as np

# matlab starts counts at 1. python starts counts at 0. hence first iteration
# will be 0, not 1 like in matlab. hence iteration+1 in code
# decay_rate in [0, 1] or (0, 1) or?
# theory?
# Sequence typing appropriate? or ndarray for all?
def update_statistics(
    mean: Sequence[float],
    covariance: np.ndarray,
    sample: Sequence[float],
    n_sample: int,
    decay_constant: float
) -> Tuple[Sequence[float], np.ndarray]:
    '''
    Update sampling statistics.

    Parameters
    ----------
    mean:
        The estimated mean of the samples, that was calculated in the previous
        iteration. It is the initial sample for first iteration.

    covariance:
        The estimated covariance matrix of the sample, that was calculated in
        the previous iteration. It is the initial estimated covariance matrix
        for the first iteration.

    sample:
        Most recent sample.

    n_sample:
        Index of current sample in the sampling iterations.

    decay_constant:
        Adaption decay, in (0, 1). Higher values result in faster decays, such
        that later iterations influence the adaption more weakly.

    Returns
    -------
    The updated values for the estimated mean and the estimated covariance
    matrix of the sample, in that order, as a tuple.
    '''
    # magic number + 1 to avoid division by zero/match matlab numbers starting
    # at 0
    update_rate = np.power(n_sample + 1, -decay_constant)

    mean = (1-update_rate)*mean + update_rate*sample

    covariance = ((1-update_rate)*covariance
                 + update_rate*np.power(sample - mean,2))

    return (mean, covariance)

# presumably covariance is always real-valued?
def is_positive_definite(matrix: np.ndarray) -> bool:
    '''
    Determines whether a matrix is positive-definite. The
    numpy.linalg.cholesky method raises a numpy.linalg.LinAlgError if its
    argument is not a Hermitian, positive-definite matrix.

    Parameters
    ----------
    matrix:
        The matrix to be checked for positive definiteness.

    Returns
    -------
    bool:
        True if the matrix is positive definite, else False.
    '''
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

# magic dividing number only different in first iteration?
# number of parameters could be determined from the shape of covariance?
# resolve meaning of MAGIC_DIVIDING_NUMBER...

# argument covariance: expects two dimensional, square, numpy.ndarray of float
#    (real-valued?)
def regularize_covariance(
    covariance: np.ndarray,
    regularization_factor: float,
    n_parameters: int,
    MAGIC_DIVIDING_NUMBER: int = 1
) -> np.ndarray:
    '''
    Regularizes the estimated covariance matrix of the sample. Useful if the
    estimated covariance matrix is ill-conditioned.

    Parameters
    ----------
    covariance:
        Estimate of the covariance matrix of the sample.

    regularization_factor:
        Larger values result in stronger regularization.

    n_parameters:
        Number of parameters in the sample.

    MAGIC_DIVIDING_NUMBER:
        magic...

    Returns
    -------
    covariance:
        Regularized estimate of the covariance matrix of the sample.
    '''
    if is_positive_definite(covariance):
        covariance += regularization_factor * np.identity(n_parameters)
        covariance = (covariance + covariance.conj().T)/2
        if is_positive_definite(covariance):
            # magic number 1000 for first iteration?
            covariance += (covariance.max()/MAGIC_DIVIDING_NUMBER
                     *np.identity(n_parameters))
            covariance = (covariance + covariance.conj().T)/2

    return covariance

# remove log_transformation_backward/forward?
# Is the description for log_acceptance in Returns correct?
def test_sample(
        log_posterior_callable: Callable,
        sample: Sequence[float],
        sample0_log_posterior: float,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        beta: float
) -> Dict:
    '''
    Determines whether to accept the proposed sample.

    Parameters
    ----------
    log_posterior_callable:
        A function that takes the proposed sample (theta_proposal) as its
        input parameter, and returns the log posterior of the sample. For
        example, it might run a simulation with the sample, then calculate
        the objective function with the simulation results and measurements.

    sample:
        Proposed sample.

    sample0_log_posterior:
        The log posterior of the (current/not proposed) sample.

    lower_bounds (respectively, upper_bounds):
        The lower (respectively, upper) bounds sample parameters.

    beta:
        Value from parallel tempering.

    Returns
    -------
    dict:
        Values are dependent upon whether the proposed sample is accepted as
        the new (alternatively, next) sample.
        If the proposed sample is accepted:
            'accepted': True.
            'log_posterior': log posterior of the proposed sample.
            'log_acceptance': a measure of the probability of accepting the
                              proposed sample.
        Else:
            'accepted': False.
            'log_acceptance': as above.
    '''
    log_acceptance = -np.inf # ???? /pAcc/
    # abuse truth value testing to check if parameters are within bounds
    if  (sum(sample < lower_bounds)
       + sum(sample > upper_bounds)) == 0:
        sample_log_posterior = log_posterior_callable(sample)
        if abs(sample_log_posterior) < np.inf:
            # what is the point of log_transformation_backward/_forward in the
            # next three lines?
            log_transformation_forward = 1 # magic number
            log_transformation_backward = 1 # magic number
            log_acceptance = (beta
                *(sample_log_posterior - sample0_log_posterior)
                + log_transformation_backward - log_transformation_forward)
            if np.isnan(log_acceptance): # possible if log_posterior_callable
                                         # has numerical issues
                log_acceptance = -np.inf
            elif log_acceptance > 0: # matlab code states "do not use min, due
                                     # to NaN behaviour in Matlab"
                log_acceptance = 0

    # here, the range is [0,1]. matlab range might be (0,1)?
    if np.log(np.random.uniform(0,1)) <= log_acceptance:
        return {
            'accepted': True,
            'log_posterior': sample_log_posterior,
            'log_acceptance': log_acceptance
        }
    else:
        return {
            'accepted': False,
            'log_acceptance': log_acceptance
        }

#consider in parent class/parallel tempering class
#
#- save results for each temperature iteration
#- allow saving for only every nth iteration
#- allow for restoration of aborted iteration

# debug values for covariance (covariance_scaling_factor, covariance_history)
# irrelevant here?
def try_sampling(
        log_posterior_callable: Callable,
        sample0: Sequence[float],
        sample0_log_posterior: float,
        covariance: np.ndarray,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        debug: bool,
        beta: float = 1
):
    '''
    Propose a new sample, and determine whether to accept the proposal.

    Parameters
    ----------
    log_posterior_callable:
        A function that takes a sample as its input parameter, and returns the
        log posterior of the sample. For example, it might run a simulation
        with the sample, then calculate the objective function with the
        simulation results and measurements.

    sample:
        Current sample.

    sample_log_posterior:
        log posterior of the current sample.

    covariance:
        Estimated covariance matrix of the sample.

    lower_bounds (respectively, upper_bounds):
        The lower (respectively, upper) bounds of the sample parameters.

    debug;
        If true, returns additional information.

    beta:
        Value from parallel tempering.

    Returns
    -------
    dict:
        'sample': Proposed sample if accepted, else current sample (sample0).
        'log_posterior': log posterior of the returned sample (above 'sample').
        'log_acceptance': a measure of the probability of accepting the
                          proposed sample.
        'accepted': True if the proposed sample is accepted, else False.
    '''
    # could report progress here

    # will return input values, if proposal is rejected
    result = {
        'sample': sample0,
        'log_posterior': sample0_log_posterior,
    }

    sample = np.random.multivariate_normal(sample0, covariance).T

    sample_proposal_result = test_sample(
            log_posterior_callable,
            sample,
            sample0_log_posterior,
            lower_bounds,
            upper_bounds,
            beta
    )

    result['log_acceptance'] = sample_proposal_result['log_acceptance']
    result['accepted'] = sample_proposal_result['accepted']

    if sample_proposal_result['accepted']:
        result['sample'] = sample
        result['log_posterior'] = sample_proposal_result['log_posterior']

    return result

# covariance_scaling_factor type? Add significance of 23% to documentation?
# replace parameter_count with len(sample)?
def estimate_covariance(
    historical_mean: Sequence[float],
    historical_covariance: np.ndarray,
    sample: Sequence[float],
    threshold_sample: int,
    decay_constant: float,
    covariance_scaling_factor,
    log_acceptance: float,
    regularization_factor: float,
    n_parameters: int,
    n_sample: int
):
    '''
    Update the estimated covariance matrix of the sample.

    Parameters
    ----------
    historical_mean:
        Estimated means of samples from all previous iterations, for all
        previous iterations.

    historical_covariance:
        Estimated covariance matrices of samples from all previous iterations,
        at each iteration.

    sample:
        Current sample.

    threshold_sample:
        Number of samples before adaption decreases significantly.
        Alternatively: a higher value reduces strong early adaption.

    decay_constant:
        Adaption decay, in (0, 1). Higher values result in faster decays, such
        that later iterations influence the adaption more weakly.

    covariance_scaling_factor:
        Scaling factor of the estimated covariance matrix, such that there is
        an overall acceptance rate of 23%.

    regularization_factor:
        Factor used to regularize the estimated covariance matrix. Larger
        values result in stronger regularization.

    n_parameters:
        Number of parameters in the sample.

    n_sample:
        Index of current sample in the sampling iterations.
    '''
    historical_mean, historical_covariance = update_statistics(
            historical_mean,
            historical_covariance,
            sample,
            max(n_sample, threshold_sample),
            decay_constant
    )

    covariance_scaling_factor = covariance_scaling_factor * np.exp(
        (np.exp(log_acceptance)-0.234) # magic number 0.234. 23.4%?
        / np.power(n_sample + 1, decay_constant)
    )

    # implemented slightly differently in matlab code
    covariance = (covariance_scaling_factor * covariance_scaling_factor
                  * historical_covariance)

    covariance = regularize_covariance(covariance,
        regularization_factor, n_parameters)

    return {
        'historical_mean': historical_mean,
        'historical_covariance': historical_covariance,
        'covariance_scaling_factor': covariance_scaling_factor,
        'covariance': covariance
    }

