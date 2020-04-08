from typing import Dict, Tuple
import numpy as np
import numbers

from ..problem import Problem
from .metropolis import MetropolisSampler


class AdaptiveMetropolisSampler(MetropolisSampler):
    """
    Metropolis-Hastings sampler with adaptive proposal covariance.
    """

    def __init__(self, options: Dict = None):
        super().__init__(options)
        self._cov = None
        self._mean_hist = None
        self._cov_hist = None
        self._cov_scale = None

    @classmethod
    def default_options(cls):
        return {
            # controls adaptation degeneration velocity of the proposals
            # in [0, 1], with 0 -> no adaptation, i.e. classical
            # Metropolis-Hastings
            'decay_constant': 0.51,
            # number of samples before adaptation decreases significantly.
            # a higher value reduces the impact of early adaptation
            'threshold_sample': 1,
            # regularization factor for ill-conditioned cov matrices of
            # the adapted proposal density. regularization might happen if the
            # eigenvalues of the cov matrix strongly differ in order
            # of magnitude. in this case, the algorithm adds a small
            # diag matrix to the cov matrix with elements of this factor
            'reg_factor': 1e-6,
            # initial covariance matrix. defaults to a unit matrix
            'cov0': None,
            # target acceptance rate
            'target_acceptance_rate': 0.234,
        }

    def initialize(self, problem: Problem, x0: np.ndarray):
        super().initialize(problem, x0)

        if self.options['cov0'] is not None:
            cov0 = self.options['cov0']
            if isinstance(cov0, numbers.Real):
                cov0 = float(cov0) * np.eye(len(x0))
        else:
            cov0 = np.eye(len(x0))
        self._cov = regularize_covariance(cov0, self.options['reg_factor'])
        self._mean_hist = self.trace_x[-1]
        self._cov_hist = self._cov
        self._cov_scale = 1.

    def _propose_parameter(self, x: np.ndarray):
        x_new = np.random.multivariate_normal(x, self._cov)
        return x_new

    def _update_proposal(self, x: np.ndarray, llh: float, log_p_acc: float,
                         n_sample_cur: int):
        # parse options
        decay_constant = self.options['decay_constant']
        threshold_sample = self.options['threshold_sample']
        reg_factor = self.options['reg_factor']
        target_acceptance_rate = self.options['target_acceptance_rate']

        # compute historical mean and covariance
        self._mean_hist, self._cov_hist = update_history_statistics(
            mean=self._mean_hist, cov=self._cov_hist, x_new=x,
            n_cur_sample=max(n_sample_cur + 1, threshold_sample),
            decay_constant=decay_constant)

        # compute covariance scaling factor
        self._cov_scale *= np.exp(
            (np.exp(log_p_acc) - target_acceptance_rate)
            / np.power(n_sample_cur + 1, decay_constant))

        # set proposal covariance
        # TODO check publication
        self._cov = self._cov_scale * self._cov_hist

        # regularize proposal covariance
        self._cov = regularize_covariance(
            cov=self._cov, reg_factor=reg_factor)


def update_history_statistics(
        mean: np.ndarray,
        cov: np.ndarray,
        x_new: np.ndarray,
        n_cur_sample: int,
        decay_constant: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update sampling statistics.

    Parameters
    ----------
    mean:
        The estimated mean of the samples, that was calculated in the previous
        iteration.
    cov:
        The estimated covariance matrix of the sample, that was calculated in
        the previous iteration.
    x_new:
        Most recent sample.
    n_cur_sample:
        Current number of samples.
    decay_constant:
        Adaption decay, in (0, 1). Higher values result in faster decays, such
        that later iterations influence the adaption more weakly.

    Returns
    -------
    mean, cov:
        The updated values for the estimated mean and the estimated covariance
        matrix of the sample.
    """
    update_rate = n_cur_sample ** (- decay_constant)

    mean = (1 - update_rate) * mean + update_rate * x_new

    dx = x_new - mean
    cov = (1 - update_rate) * cov + \
        update_rate * dx.reshape((-1, 1)) @ dx.reshape((1, -1))

    return mean, cov


def regularize_covariance(cov: np.ndarray, reg_factor: float):
    """
    Regularize the estimated covariance matrix of the sample. Useful if the
    estimated covariance matrix is ill-conditioned.

    Parameters
    ----------
    cov:
        Estimate of the covariance matrix of the sample.
    reg_factor:
        Regularization factor. Larger values result in stronger regularization.

    Returns
    -------
    cov:
        Regularized estimate of the covariance matrix of the sample.
    """
    n_par = cov.shape[0]
    if not is_positive_definite(cov):
        # add regularization factor
        cov += reg_factor * np.eye(n_par)
        # make symmetric
        cov = (cov + cov.T) / 2
        if not is_positive_definite(cov):
            # add strong regularization factor
            cov += cov.max() * np.eye(n_par)
            # make symmetric
            cov = (cov + cov.T) / 2

    return cov


def is_positive_definite(matrix: np.ndarray) -> bool:
    """Determines whether a matrix is positive-definite.

    Parameters
    ----------
    matrix:
        The matrix to be checked for positive definiteness.

    Returns
    -------
    bool:
        True if the matrix is positive definite, else False.
    """
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True
