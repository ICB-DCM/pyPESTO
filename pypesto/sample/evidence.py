"""Various methods for estimating the log evidence of a model."""

import logging
from typing import Optional, Union

import numpy as np
from scipy import stats
from scipy.integrate import simpson, trapezoid
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ..C import SIMPSON, STEPPINGSTONE, TRAPEZOID
from ..objective import (
    AggregatedObjective,
    NegLogParameterPriors,
    NegLogPriors,
)
from ..problem import Problem
from ..result import Result
from .diagnostics import geweke_test

logger = logging.getLogger(__name__)


def laplace_approximation_log_evidence(
    problem: Problem, x: np.ndarray
) -> float:
    """
    Compute the log evidence using the Laplace approximation.

    The objective in your `problem` must be a negative log posterior, and support Hessian computation.

    Parameters
    ----------
    problem:
        The problem to compute the log evidence for.
    x:
        The maximum a posteriori estimate at which to compute the log evidence.

    Returns
    -------
    log_evidence: float
    """
    hessian = problem.objective(
        problem.get_reduced_vector(x), sensi_orders=(2,)
    )
    _, log_det = np.linalg.slogdet(hessian)
    log_prop_posterior = -problem.objective(problem.get_reduced_vector(x))
    log_evidence = (
        0.5 * np.log(2 * np.pi) * len(problem.x_free_indices)
        - 0.5 * log_det
        + log_prop_posterior
    )
    return log_evidence


def harmonic_mean_log_evidence(
    result: Result,
    prior_samples: Optional[np.ndarray] = None,
    neg_log_likelihood_fun: Optional[callable] = None,
) -> float:
    """
    Compute the log evidence using the harmonic mean estimator.

    Stabilized harmonic mean estimator is used if prior samples are provided.
    Newton and Raftery (1994): https://doi.org/10.1111/j.2517-6161.1994.tb01956.x

    Parameters
    ----------
    result:
    prior_samples:
        Samples from the prior distribution. If samples from the prior are provided,
        the stabilized harmonic mean is computed (recommended). Then, the likelihood function must be provided as well.
    neg_log_likelihood_fun: callable
        Function to evaluate the negative log likelihood. Necessary if prior_samples is not `None`.

    Returns
    -------
    log_evidence
    """
    if result.sample_result is None:
        raise ValueError("No samples available. Run sampling first.")

    # compute negative log likelihood from traces
    burn_in = geweke_test(result)
    trace_neglogpost = result.sample_result.trace_neglogpost[0, burn_in:]
    trace_neglogprior = result.sample_result.trace_neglogprior[0, burn_in:]
    neg_log_likelihoods_posterior = trace_neglogpost - trace_neglogprior

    if prior_samples is None:
        # compute harmonic mean from samples
        return -logsumexp(neg_log_likelihoods_posterior) + np.log(
            neg_log_likelihoods_posterior.size
        )

    # compute stabilized harmonic mean
    if prior_samples is not None and neg_log_likelihood_fun is None:
        raise ValueError(
            "you need to provide a likelihood function to evaluate prior samples"
        )

    # compute delta (ratio of prior to posterior samples)
    n_samples_prior = len(prior_samples)
    n_samples_posterior = len(trace_neglogpost)
    delta = n_samples_prior / (n_samples_prior + n_samples_posterior)
    neg_log_likelihoods_prior = np.array(
        [neg_log_likelihood_fun(x) for x in prior_samples]
    )
    log_likelihoods_stack = -np.concatenate(
        [neg_log_likelihoods_prior, neg_log_likelihoods_posterior]
    )

    def _log_evidence_objective(log_p: float):
        # Helper function to compute the log evidence with stabilized harmonic mean
        log_w_i = logsumexp(
            np.stack(
                (
                    log_p * np.ones_like(log_likelihoods_stack),
                    log_likelihoods_stack,
                ),
                axis=1,
            ),
            b=np.array([delta, 1 - delta]),
            axis=1,
        )
        res, sign = logsumexp(
            [
                log_p,
                logsumexp(log_likelihoods_stack - log_w_i)
                - logsumexp(-log_w_i),
            ],
            b=[1, -1],
            return_sign=True,
        )
        return sign * res

    sol = minimize_scalar(_log_evidence_objective)
    return sol.x


def parallel_tempering_log_evidence(
    result: Result,
    method: str = "trapezoid",
    use_all_chains: bool = True,
) -> Union[float, None]:
    """Perform thermodynamic integration or steppingstone sampling to estimate the log evidence.

    Thermodynamic integration is performed by integrating the mean log likelihood over the temperatures.
    Errors might come from the samples itself or the numerical integration.
    Steppingstone sampling is a form of importance sampling that uses the maximum likelihood of each temperature.
    It does not require an integration, but can be biased for a small number of temperatures.
    See (Annis et al., 2019), https://doi.org/10.1016/j.jmp.2019.01.005, for more details.

    This should be used with a beta decay temperature schedule and not with the adaptive version of
     parallel tempering sampling as the temperature schedule is not optimal for thermodynamic integration.

    Parameters
    ----------
    result:
        Result object containing the samples.
    method:
        Integration method, either 'trapezoid' or 'simpson' to perform thermodynamic integration
        (uses scipy for integration) or 'steppingstone' to perform steppingstone sampling.
    use_all_chains:
        If True, calculate burn-in for each chain and use the maximal burn-in for all chains for the integration.
        This will fail if not all chains have converged yet.
        Otherwise, use only the converged chains for the integration (might increase the integration error).
    """
    # compute burn in for all chains but the last one (prior only)
    burn_ins = np.zeros(len(result.sample_result.betas), dtype=int)
    for i_chain in range(len(result.sample_result.betas)):
        burn_ins[i_chain] = geweke_test(result, chain_number=i_chain)
    max_burn_in = int(np.max(burn_ins))

    if max_burn_in >= result.sample_result.trace_x.shape[1]:
        logger.warning(
            f"At least {np.sum(burn_ins >= result.sample_result.trace_x.shape[1])} chains seem to not have "
            f"converged yet. You may want to use a larger number of samples."
        )
        if use_all_chains:
            raise ValueError(
                "Not all chains have converged yet. You may want to use a larger number of samples, "
                "or try ´use_all_chains=False´, which might increase the integration error."
            )

    if use_all_chains:
        # estimate mean of log likelihood for each beta
        trace_loglike = (
            result.sample_result.trace_neglogprior[::-1, max_burn_in:]
            - result.sample_result.trace_neglogpost[::-1, max_burn_in:]
        )
        mean_loglike_per_beta = np.mean(trace_loglike, axis=1)
        temps = result.sample_result.betas[::-1]
    else:
        # estimate mean of log likelihood for each beta if chain has converged
        mean_loglike_per_beta = []
        trace_loglike = []
        temps = []
        for i_chain in reversed(range(len(result.sample_result.betas))):
            if burn_ins[i_chain] < result.sample_result.trace_x.shape[1]:
                # save temperature-chain as it is converged
                temps.append(result.sample_result.betas[i_chain])
                # calculate mean log likelihood for each beta
                trace_loglike_i = (
                    result.sample_result.trace_neglogprior[
                        i_chain, burn_ins[i_chain] :
                    ]
                    - result.sample_result.trace_neglogpost[
                        i_chain, burn_ins[i_chain] :
                    ]
                )
                trace_loglike.append(trace_loglike_i)
                mean_loglike_per_beta.append(np.mean(trace_loglike_i))

    if method == TRAPEZOID:
        log_evidence = trapezoid(
            # integrate from low to high temperature
            y=mean_loglike_per_beta,
            x=temps,
        )
    elif method == SIMPSON:
        log_evidence = simpson(
            # integrate from low to high temperature
            y=mean_loglike_per_beta,
            x=temps,
        )
    elif method == STEPPINGSTONE:
        log_evidence = steppingstone(temps=temps, trace_loglike=trace_loglike)
    else:
        raise ValueError(
            f"Unknown method {method}. Choose 'trapezoid', 'simpson' for thermodynamic integration or ",
            "'steppingstone' for steppingstone sampling.",
        )

    return log_evidence


def steppingstone(temps: np.ndarray, trace_loglike: np.ndarray) -> float:
    """Perform steppingstone sampling to estimate the log evidence.

    Implementation based on  Annis et al. (2019): https://doi.org/10.1016/j.jmp.2019.01.005.

    Parameters
    ----------
    temps:
        Temperature values.
    trace_loglike:
        Log likelihood values for each temperature.
    """
    from scipy.special import logsumexp

    ss_log_evidences = np.zeros(len(temps) - 1)
    for t_i in range(1, len(temps)):
        # we use the maximum likelihood times the temperature difference to stabilize the logsumexp
        # original formulation uses only the maximum likelihood, this is equivalent
        ss_log_evidences[t_i - 1] = logsumexp(
            trace_loglike[t_i - 1] * (temps[t_i] - temps[t_i - 1])
        ) - np.log(trace_loglike[t_i - 1].size)
    log_evidence = np.sum(ss_log_evidences)
    return log_evidence


def bridge_sampling_log_evidence(
    result: Result,
    n_posterior_samples_init: Optional[int] = None,
    initial_guess_log_evidence: Optional[float] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> float:
    """
    Compute the log evidence using bridge sampling.

    Based on "A Tutorial on Bridge Sampling" by Gronau et al. (2017): https://doi.org/10.1016/j.jmp.2017.09.005.
    Using the optimal bridge function by Meng and Wong (1996) which minimises the relative mean-squared error.
    Proposal function is calibrated using posterior samples, which are not used for the final bridge estimate
    (as this may result in an underestimation of the marginal likelihood, see Overstall and Forster (2010)).

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    n_posterior_samples_init:
        Number of samples used to calibrate the proposal function. By default, half of the posterior samples are used.
    initial_guess_log_evidence:
        Initial guess for the log evidence. By default, the Laplace approximation is used to compute the initial guess.
    max_iter:
        Maximum number of iterations. Default is 1000.
    tol:
        Tolerance for convergence. Default is 1e-6.


    Returns
    -------
    log_evidence
    """
    if result.sample_result is None:
        raise ValueError("No samples available. Run sampling first.")
    if not isinstance(result.problem.objective, AggregatedObjective):
        raise ValueError("Objective must be an AggregatedObjective.")

    # use Laplace approximation to get initial guess for p(y)
    if initial_guess_log_evidence is None:
        initial_guess_log_evidence = laplace_approximation_log_evidence(
            problem=result.problem, x=result.optimize_result.x[0]
        )
    # extract posterior samples
    burn_in = geweke_test(result)
    posterior_samples = result.sample_result.trace_x[0, burn_in:]

    # build proposal function from posterior samples
    if n_posterior_samples_init is None:
        n_posterior_samples_init = int(posterior_samples.shape[0] * 0.5)
    # randomly select samples for calibration
    calibration_index = np.random.choice(
        np.arange(posterior_samples.shape[0]),
        n_posterior_samples_init,
        replace=False,
    )
    samples_calibration = posterior_samples[calibration_index]
    # remove calibration samples from posterior samples
    posterior_samples = posterior_samples[
        [
            j
            for j in range(posterior_samples.shape[0])
            if j not in calibration_index
        ]
    ]
    # generate proposal samples and define proposal function
    n_proposal_samples = posterior_samples.shape[0]
    posterior_mean = np.mean(samples_calibration, axis=0)
    posterior_cov = np.cov(samples_calibration.T)
    # if covariance matrix is not positive definite (numerically), use diagonal covariance matrix only
    try:
        # proposal density function
        log_proposal_fun = stats.multivariate_normal(
            mean=posterior_mean, cov=posterior_cov
        ).logpdf
    except np.linalg.LinAlgError:
        posterior_cov = np.diag(np.diag(posterior_cov))
        log_proposal_fun = stats.multivariate_normal(
            mean=posterior_mean, cov=posterior_cov
        ).logpdf

    # generate proposal samples
    if posterior_cov.size == 1:
        # univariate case
        proposal_samples = np.random.normal(
            loc=posterior_mean,
            scale=np.sqrt(posterior_cov),
            size=n_proposal_samples,
        )
        proposal_samples = proposal_samples.reshape(-1, 1)
    else:
        # multivariate case
        proposal_samples = np.random.multivariate_normal(
            mean=posterior_mean, cov=posterior_cov, size=n_proposal_samples
        )

    # Compute the weights for the bridge sampling estimate
    log_s1 = np.log(
        posterior_samples.shape[0]
        / (posterior_samples.shape[0] + n_proposal_samples)
    )
    log_s2 = np.log(
        n_proposal_samples / (posterior_samples.shape[0] + n_proposal_samples)
    )

    # Start with the initial guess for p(y)
    log_p_y = initial_guess_log_evidence

    # Compute the log-likelihood, log-prior, and log-proposal for the posterior and proposal samples
    # assumes that the objective function is the negative log-likelihood + negative log-prior

    # get index of prior in the objective function
    likelihood_fun_indices = []
    for i, obj in enumerate(result.problem.objective._objectives):
        if not isinstance(obj, NegLogParameterPriors) and not isinstance(
            obj, NegLogPriors
        ):
            likelihood_fun_indices.append(i)

    def log_likelihood_fun(x_array):
        return np.array(
            [
                np.sum(
                    [
                        -obj(
                            result.problem.get_full_vector(
                                x=x, x_fixed_vals=result.problem.x_fixed_vals
                            )
                        )
                        for obj_i, obj in enumerate(
                            result.problem.objective._objectives
                        )
                        if obj_i in likelihood_fun_indices
                    ]
                )
                for x in x_array
            ]
        )

    def log_prior_fun(x_array):
        return np.array(
            [
                np.sum(
                    [
                        -obj(
                            result.problem.get_full_vector(
                                x=x, x_fixed_vals=result.problem.x_fixed_vals
                            )
                        )
                        for obj_i, obj in enumerate(
                            result.problem.objective._objectives
                        )
                        if obj_i not in likelihood_fun_indices
                    ]
                )
                for x in x_array
            ]
        )

    log_likelihood_posterior = log_likelihood_fun(posterior_samples)
    log_prior_posterior = log_prior_fun(posterior_samples)
    log_proposal_posterior = log_proposal_fun(posterior_samples)

    log_likelihood_proposal = log_likelihood_fun(proposal_samples)
    log_prior_proposal = log_prior_fun(proposal_samples)
    log_proposal_proposal = log_proposal_fun(proposal_samples)

    log_h_posterior_1 = log_s1 + log_likelihood_posterior + log_prior_posterior
    log_h_proposal_1 = log_s1 + log_likelihood_proposal + log_prior_proposal
    for i in range(max_iter):
        # Compute h(θ) for posterior samples
        log_h_posterior_2 = log_s2 + log_p_y + log_proposal_posterior
        log_h_posterior = logsumexp([log_h_posterior_1, log_h_posterior_2])

        # Compute h(θ) for proposal samples
        log_h_proposal_2 = log_s2 + log_p_y + log_proposal_proposal
        log_h_proposal = logsumexp([log_h_proposal_1, log_h_proposal_2])

        # Calculate the numerator and denominator for the bridge sampling estimate
        temp = log_likelihood_proposal + log_prior_proposal + log_h_proposal
        log_numerator = logsumexp(temp) - np.log(
            temp.size
        )  # compute mean in log space
        temp = log_proposal_posterior + log_h_posterior
        log_denominator = logsumexp(temp) - np.log(
            temp.size
        )  # compute mean in log space

        # Update p(y)
        log_p_y_new = log_numerator - log_denominator

        # Check for convergence
        if abs(log_p_y_new - log_p_y) < tol:
            break

        log_p_y = log_p_y_new

        if i == max_iter - 1:
            logger.warning(
                "Bridge sampling did not converge in the given number of iterations."
            )

    return log_p_y
