from typing import Callable, Sequence, Tuple
import numpy as np
# import math
# import random

# matlab starts counts at 1. python starts counts at 0. hence first iteration
# will be 0, not 1 like in matlab. hence iteration+1 in code
# decay_rate in [0, 1] or (0, 1) or?
# theory?
# Sequence typing appropriate? or ndarray for all?
def update_statistics(
    mean0: Sequence[float],
    covariance0: np.ndarray,
    theta: Sequence[float],
    iteration: int,
    decay_rate: float
) -> Tuple[Sequence[float], np.ndarray]:
    '''
    Update sampling statistics.
    
    Parameters
    ----------
    mean0:
        The estimated mean of the samples, that was calculated in the previous
        iteration. It is the initial sample for first iteration.

    covariance0:
        The estimated covariance matrix of the sample, that was calculated in
        the previous iteration. It is the initial estimated covariance matrix
        for the first iteration.

    theta:
        Most recent sample.

    iteration:
        Index of current sample in the sampling iterations.

    decay_rate:
        Adaption decay, in (0, 1). Higher values result in faster decays, such
        that later iterations influence the adaption more weakly.

    Returns
    -------
    The updated values for the estimated mean and the estimated covariance
    matrix of the sample, in that order, as a tuple.
    '''
    # magic number + 1 to avoid division by zero/match matlab numbers starting
    # at 0
    update_rate = np.power(iteration + 1, -decay_rate)
    mean = (1-update_rate)*mean0 + update_rate*theta
    covariance = ((1-update_rate)*covariance0
                 + update_rate*(theta - mean)*(theta - mean))
    return (mean, covariance)

# presumably covariance is always real-valued?
def is_positive_definite(matrix: np.ndarray) -> bool:
    '''
    Determines whether a matrix is positive-definite. The
    numpy.linalg.cholesky method raises a numpy.linalg.LinAlgError if its
    input is not a Hermitian, positive-definite matrix.

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
def adaptive_metropolis_regularizer(
    covariance: np.ndarray,
    regularization_factor: float,
    parameter_count: int,
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
    
    parameter_count:
        Number of parameters in the sample.

    Returns
    -------
    covariance:
        Regularized estimate of the covariance matrix of the sample.
    '''
    if is_positive_definite(covariance):
        covariance += regularization_factor * np.identity(parameter_count)
        covariance = (covariance + covariance.conj().T)/2
        if is_positive_definite(covariance):
            # magic number 1000 for first iteration?
            covariance += (covariance.max()/MAGIC_DIVIDING_NUMBER
                     *np.identity(parameter_count))
            covariance = (covariance + covariance.conj().T)/2

    return covariance

# remove log_transformation_backward/forward?
# Is the description for log_acceptance in Returns correct?
def adaptive_metropolis_proposal_accepted(
    log_posterior_callable: Callable,
    theta_proposal: Sequence[float],
    log_posterior_theta0: float,
    theta_bounds_lower: Sequence[float],
    theta_bounds_upper: Sequence[float]
):
    '''
    Determines whether to accept the proposed sample.

    Parameters
    ----------
    log_posterior_callable:
        A function that takes the proposed sample (theta_proposal) as its
        input parameter, and returns the log posterior of the sample. For
        example, it might run a simulation with the sample, then calculate
        the objective function with the simulation results and measurements.

    theta_proposal:
        Proposed sample.

    log_posterior_theta0:
        The log posterior of the current sample.

    theta_bounds_lower (respectively, theta_bounds_upper):
        The lower (respectively, upper) bounds of the parameters in the
        sample.

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
    if  (sum(theta_proposal < theta_bounds_lower)
       + sum(theta_proposal > theta_bounds_upper)) == 0:
        log_posterior_proposal = log_posterior_callable(theta_proposal)
        if abs(log_posterior_proposal) < np.inf:
            # what is the point of log_transformation_backward/_forward in the
            # next three lines?
            log_transformation_forward = 1 # magic number
            log_transformation_backward = 1 # magic number
            log_acceptance = (log_posterior_proposal - log_posterior_theta0
                + log_transformation_backward - log_transformation_forward)
            if np.isnan(log_acceptance): # possible if log_posterior_callable
                                         # has numerical issues
                log_acceptance = -np.inf
            elif log_acceptance > 0: # matlab code states "do not use min, due
                                     # to NaN behaviour in Matlab"
                log_acceptance = 0

    # here, the range is [0,1]. matlab range might be (0,1)?
    if np.less_equal(np.log(float(np.random.uniform(0, 1, 1))), log_acceptance):
        return {
            'accepted': True,
            'log_posterior': log_posterior_proposal,
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
def adaptive_metropolis_update_theta(
    log_posterior_callable: Callable,
    theta: Sequence[float],
    log_posterior_theta: float,
    covariance: np.ndarray,
    theta_bounds_lower: Sequence[float],
    theta_bounds_upper: Sequence[float],
    debug: bool
):
    '''
    Propose a new sample, and determine whether to accept the proposal.

    Parameters
    ----------
    log_posterior_callable:
        A function that takes the proposed sample (theta_proposal) as its
        input parameter, and returns the log posterior of the sample. For
        example, it might run a simulation with the sample, then calculate the
        objective function with the simulation results and measurements.

    theta:
        Current sample.

    log_posterior_theta:
        log posterior of the current sample.

    covariance:
        Estimated covariance matrix of the sample.
        
    theta_bounds_lower (respectively, theta_bounds_upper):
        The lower (respectively, upper) bounds of the parameters.

    debug;
        If true, returns additional information.

    Returns
    -------
    dict:
        'theta': Proposed sample if accepted (theta_proposal), else current
                 sample (theta).
        'log_posterior': log posterior of the returned sample (above 'theta').
        'log_acceptance': a measure of the probability of accepting the
                          proposed sample.
        'accepted': True if the proposed sample is accepted, else False.
    '''
    # could report progress here

    # will return input values, if proposal is rejected
    result = {
        'theta': theta,
        'log_posterior': log_posterior_theta,
    }

    theta_proposal = np.random.multivariate_normal(theta, covariance).T

    theta_proposal_result = adaptive_metropolis_proposal_accepted(
        log_posterior_callable,
        theta_proposal,
        log_posterior_theta,
        theta_bounds_lower,
        theta_bounds_upper
    )

    result['log_acceptance'] = theta_proposal_result['log_acceptance']
    result['accepted'] = theta_proposal_result['accepted']

    if theta_proposal_result['accepted']:
        result['theta'] = theta_proposal
        result['log_posterior'] = theta_proposal_result['log_posterior']

    return result

# covariance_scaling_factor type? Add significance of 23% to documentation?
# replace parameter_count with len(theta)?
def adaptive_metropolis_update_covariance(
    historical_mean: np.ndarray,
    historical_covariance: np.ndarray,
    theta: Sequence[float],
    threshold_iteration: int,
    decay_rate: float,
    covariance_scaling_factor,
    log_acceptance: float,
    regularization_factor: float,
    parameter_count: int,
    iteration: int
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

    theta:
        Current sample.

    threshold_iteration:
        Number of iterations before adaption decreases significantly.
        Alternatively: a higher value reduces strong early adaption.

    decay_rate:
        Adaption decay, in (0, 1). Higher values result in faster decays, such
        that later iterations influence the adaption more weakly.

    covariance_scaling_factor:
        Scaling factor of the estimated covariance matrix, such that there is
        an overall acceptance rate of 23%.

    regularization_factor:
        Factor used to regularize the estimated covariance matrix. Larger
        values result in stronger regularization.
    
    parameter_count:
        Number of parameters in the sample.

    iteration:
        Index of current sample in the sampling iterations.
    '''
    historical_mean, historical_covariance = update_statistics(
        historical_mean, historical_covariance, theta,
        max(iteration, threshold_iteration), decay_rate)

    covariance_scaling_factor = covariance_scaling_factor * np.exp(
        (np.exp(log_acceptance)-0.234) # magic number 0.234. 23.4%?
        / np.power(iteration+1, decay_rate)
    )

    # implemented slightly differently in matlab code
    covariance = (covariance_scaling_factor * covariance_scaling_factor
                  * historical_covariance)

    covariance = adaptive_metropolis_regularizer(covariance,
        regularization_factor, parameter_count)

    return {
        'historical_mean': historical_mean,
        'historical_covariance': historical_covariance,
        'covariance_scaling_factor': covariance_scaling_factor,
        'covariance': covariance
    }

# add types of arguments
#options is dictionary (in absence of SamplerOptions object)
#returns dictionary (in absence of SamplerResults object)

# should accepted_count start at 1 as, in
# `result['cumulative_chain_acceptance_rate'][i] = 100*accepted_count/(i+1)`,
# 1 is added to i to avoid division by zero? Seems like it's better at = 0,
# then `result['cumulative_chain_acceptance_rate'] is an actual percentage
def adaptive_metropolis(
    log_posterior_callable: Callable,
    theta: Sequence[float], # could specify numpy array
    options: dict
) -> dict:
    '''
    Generate samples of parameter values using the adaptive
    Metropolis-Hastings algorithm.

    Parameters
    ----------
    log_posterior_callable:
        A function that takes the proposed sample (theta_proposal) as its
        input parameter, and returns the log posterior of the sample. For
        example, it might run a simulation with the sample, then calculate the
        objective function with the simulation results and measurements.

    theta:
        Initial sample of parameters.

    options:
        Dictionary, with keys described below.

    Options
    -------
    debug:
        Return additional information if True. 

    covariance:
        Estimate of the covariance matrix of the initial sample of parameters.

    theta_bounds_lower (respectively, theta_bounds_upper):
        The lower (respectively, upper) bounds of the parameters in the
        sample.

    iterations:
        Number of samples to be generated.
        
    decay_rate:
        Adaption decay, in (0, 1). Higher values result in faster decays, such
        that later iterations influence the adaption more weakly.

    threshold_iteration:
        Number of iterations before adaption decreases significantly.
        Alternatively: a higher value reduces strong early adaption.

    regularization_factor:
        Factor used to regularize the estimated covariance matrix. Larger
        values result in stronger regularization.

    Returns
    -------
    result:
        Dictionary, with keys described below.

    Result
    ------
    theta:
        All generated samples.

    log_posterior:
        log posterior of all generated samples.

    (if debug) cumulative_chain_acceptance_rate:
        The percentage of samples that were accepted, at each sampling.

    (if debug) covariance_scaling_factor:
        Scaling factor of the estimated covariance matrix, such that there is
        an overall acceptance rate of 23%, for each sample.
        
    (if debug) historical_covariance:
        Estimated covariance matrices of samples from all previous iterations,
        at each iteration.
    '''

    # boolean, whether to collect debug information /doDebug/
    debug = options['debug']
    # initial proposal(?) covariance matrix of parameters /sigma0/
    covariance = options['covariance']
    # lower bounds of parameters /thetaMin/
    theta_bounds_lower = options['theta_bounds_lower']
    # upper bounds of parameters /thetaMax/
    theta_bounds_upper = options['theta_bounds_upper']
    parameter_count = len(theta)  # number of parameters /nPar/

    # desired number of sampling iterations /nIter/
    iterations = options['iterations']
    # adapting decay, (0,1) or [0,1]? /alpha/
    decay_rate = options['decay_rate']
    # adaption control /memoryLength/
    threshold_iteration = options['threshold_iteration']
    # regularization factor /regFactor/
    regularization_factor = options['regularization_factor']

    result = {}
    result['theta'] = np.full([parameter_count, iterations], np.nan)
    result['log_posterior'] = np.full([iterations], np.nan)
    if debug:
        for s in [
            'cumulative_chain_acceptance_rate', #  /res.acc/
            'covariance_scaling_factor',  # /res.sigmaScale/
        ]:
            result[s] = np.full([iterations], np.nan)
        result['historical_covariance'] = np.full([*covariance.shape,
            iterations], np.nan)  # /res.sigmaHist/

    accepted_count = 0
    covariance_scaling_factor = 1.0 # /sigmaScale/

    historical_mean = theta # /muHist/
    historical_covariance = covariance #? /sigmaHist/

    covariance = adaptive_metropolis_regularizer(covariance,
        regularization_factor, parameter_count, MAGIC_DIVIDING_NUMBER = 1000)

    log_posterior_theta = log_posterior_callable(theta) # /logPost/

    for i in range(iterations):
        theta_update_result = adaptive_metropolis_update_theta(
            log_posterior_callable, theta, log_posterior_theta, covariance,
            theta_bounds_lower, theta_bounds_upper, debug)
        theta = theta_update_result['theta']
        log_posterior_theta = theta_update_result['log_posterior']
        if theta_update_result['accepted']:
            accepted_count += 1

        covariance_update_result = adaptive_metropolis_update_covariance(
            historical_mean, historical_covariance, theta,
            threshold_iteration, decay_rate, covariance_scaling_factor,
            theta_update_result['log_acceptance'], regularization_factor,
            parameter_count, i)

        historical_mean = covariance_update_result['historical_mean']
        historical_covariance = (
            covariance_update_result['historical_covariance'])
        covariance_scaling_factor = (
            covariance_update_result['covariance_scaling_factor'])
        covariance = covariance_update_result['covariance']

        result['theta'][:,i] = theta
        result['log_posterior'][i] = log_posterior_theta
        if debug:
            # magic number 1 here to avoid division by zero. not an issue in
            # Matlab, as i > 0 there
            result['cumulative_chain_acceptance_rate'][i] = (
                    100*accepted_count/(i+1))
            result['covariance_scaling_factor'][i] = covariance_scaling_factor
            result['historical_covariance'][..., i] = historical_covariance

    return result
