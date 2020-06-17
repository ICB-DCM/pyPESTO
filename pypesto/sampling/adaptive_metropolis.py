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
            # show progress
            'show_progress': True,
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

    def _update_proposal(self, x: np.ndarray, lpost: float, log_p_acc: float,
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
        Adaption decay, in [0, 1]. Higher values result in faster decays, such
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


def regularize_covariance(cov: np.ndarray, reg_factor: float) -> np.ndarray:
    """
    Regularize the estimated covariance matrix of the sample. Useful if the
    estimated covariance matrix is ill-conditioned.
    Increments the diagonal a little to ensure positivity.

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
    eig = np.linalg.eigvals(cov)
    eig_min = min(eig)
    if eig_min <= 0:
        cov += (abs(eig_min) + reg_factor) * np.eye(cov.shape[0])
    return cov
