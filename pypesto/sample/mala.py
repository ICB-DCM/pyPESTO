import numpy as np

from ..problem import Problem
from .adaptive_metropolis import AdaptiveMetropolisSampler


class MalaSampler(AdaptiveMetropolisSampler):
    """Metropolis-Adjusted Langevin Algorithm (MALA) sampler with preconditioning.

    MALA is a gradient-based MCMC method that uses the gradient of the
    log-posterior to guide the proposal distribution. This allows for
    more efficient exploration of the parameter space compared to
    standard random-walk Metropolis-Hastings.

    The proposal distribution is:
    x_new = x + (step_size / 2) * M * grad_log_p(x) + sqrt(step_size) * sqrt(M) * noise

    where grad_log_p(x) is the gradient of the log-posterior at x,
    M is the preconditioning matrix, step_size is the discretization step size,
    and noise is standard Gaussian noise.

    Preconditioning can significantly improve convergence for poorly conditioned
    problems by rescaling the parameter space. Here, we use an adaptive covariance
    matrix as the preconditioning matrix M, which is updated during sampling.
    We aim to converge to a fixed target acceptance rate of 0.574, as suggested
    by theoretical results for MALA.

    By default, the step size is set to sqrt((1/dim)^(1/3)), where dim is the
    dimension of the target distribution, following Boisvert-Beaudry and Bedard (2022).
    This scaling helps maintain efficient exploration as the dimensionality increases,
    and leads to an asymptotically performant sampler, in the sense that exploring the
    parameter space requires O(dim^(1/3)) steps.

    For reference, see:
    * Roberts et al. 1996.
      Exponential convergence of Langevin distributions and their
      discrete approximations
      (https://doi.org/10.2307/3318418)
    * Girolami & Calderhead 2011.
      Riemann manifold Langevin and Hamiltonian Monte Carlo methods
      (https://doi.org/10.1111/j.1467-9868.2010.00765.x)
    * Boisvert-Beaudry and Bedard 2022.
      MALA with annealed proposals: a generalization of locally and globally balanced proposal distributions
    """

    def __init__(self, step_size: float = None, options: dict = None):
        """Initialize the sampler.

        Parameters
        ----------
        step_size:
            Step size for the Langevin dynamics. If None, it will be set to
            sqrt((1/n_parameter)^(1/3)) during initialization.
        options:
            Options for the sampler.
        """
        super().__init__(options)
        self.step_size = step_size
        self._dimension_of_target_weight = 1.0

    @classmethod
    def default_options(cls):
        """Return the default options for the sampler."""
        return {
            # controls adaptation degeneration velocity of the proposals
            # in [0, 1], with 0 -> no adaptation, i.e. classical
            # Metropolis-Hastings
            "decay_constant": 0.51,
            # number of samples before adaptation decreases significantly.
            # a higher value reduces the impact of early adaptation
            "threshold_sample": 1,
            # regularization factor for ill-conditioned cov matrices of
            # the adapted proposal density.
            "reg_factor": 1e-8,
            # maximum number of attempts to regularize the covariance matrix
            "max_tries": 10,
            # initial covariance matrix. defaults to a unit matrix
            "cov0": None,
            # target acceptance rate
            "target_acceptance_rate": 0.574,
            # show progress
            "show_progress": None,
        }

    def initialize(self, problem: Problem, x0: np.ndarray):
        """Initialize the sampler."""
        # Check if gradient is available
        if not problem.objective.has_grad:
            raise ValueError("MALA sampler requires gradient information.")

        # Boisvert-Beaudry and Bedard (2022)
        # Set step size to problem dimension if not specified
        self.step_size = (
            np.sqrt((1 / problem.dim) ** (1 / 3))
            if self.step_size is None
            else self.step_size
        )
        self._dimension_of_target_weight = 1 + (1 / problem.dim) ** (1 / 3)

        super().initialize(problem, x0)
        self._current_x_grad = -self.neglogpost.get_grad(x0)

    def _propose_parameter(self, x: np.ndarray):
        """Propose a parameter using preconditioned MALA dynamics."""
        # Get gradient of log-posterior at current position
        grad = -self.neglogpost.get_grad(x)

        # Apply preconditioning to gradient
        precond_grad = self._cov @ grad

        # Generate standard Gaussian noise
        noise = np.random.randn(len(x))

        # Apply preconditioning to noise (via Cholesky decomposition)
        precond_noise = self._cov_chol @ noise

        # Preconditioned MALA proposal: x + (h/2) * M * grad + sqrt(h) * sqrt(M) * noise
        drift = (self.step_size / 2.0) * precond_grad
        diffusion = np.sqrt(self.step_size) * precond_noise

        x_new: np.ndarray = (
            x + self._dimension_of_target_weight * drift + diffusion
        )
        return x_new, grad

    def _compute_transition_log_prob(
        self, x_from: np.ndarray, x_to: np.ndarray, x_from_grad: np.ndarray
    ):
        """Compute the log probability of transitioning from x_from to x_to with preconditioning."""
        # Apply preconditioning to gradient
        precond_grad_from = self._cov @ x_from_grad

        # Mean of the preconditioned proposal distribution
        mean = (
            x_from
            + self._dimension_of_target_weight
            * (self.step_size / 2.0)
            * precond_grad_from
        )

        # For preconditioned MALA, the covariance is step_size * M
        # We need to compute the log probability under N(mean, step_size * M)
        diff = x_to - mean

        # Use Cholesky decomposition for efficient computation
        # We need to solve L @ L^T @ z = diff, i.e., z = M^{-1} @ diff
        # This is done by first solving L @ y = diff, then L^T @ z = y
        try:
            # Forward substitution: L @ y = diff
            y = np.linalg.solve(self._cov_chol, diff)
            # Quadratic form: diff^T @ M^{-1} @ diff = y^T @ y
            log_prob = -0.5 * np.dot(y, y) / self.step_size
        except np.linalg.LinAlgError:
            # If matrix is singular, return -inf
            return -np.inf

        # Add normalization constant: -0.5 * log|2π * step_size * M|
        # = -0.5 * (d * log(2π * step_size) + log|M|)
        # log|M| = 2 * log|L| = 2 * sum(log(diag(L)))
        d = len(x_from)
        log_det_cov = 2 * np.sum(np.log(np.diag(self._cov_chol)))
        log_prob -= 0.5 * (
            d * np.log(2 * np.pi * self.step_size) + log_det_cov
        )

        return log_prob
