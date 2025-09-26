"""Utility functions and constants for tests. Mainly problem definitions."""

import numpy as np
import scipy.optimize as so
from scipy.stats import multivariate_normal, norm, uniform

import pypesto

# Constants for Gaussian problems or Uniform with Gaussian prior
MU = 0  # Gaussian mean
SIGMA = 1  # Gaussian standard deviation
LB_GAUSSIAN = [-10]  # Lower bound for Gaussian problem
UB_GAUSSIAN = [10]  # Upper bound for Gaussian problem
LB_GAUSSIAN_MODES = [-100]  # Lower bound for Gaussian modes problem
UB_GAUSSIAN_MODES = [200]  # Upper bound for Gaussian modes problem
X_NAMES = ["x"]  # Parameter names
MIXTURE_WEIGHTS = [0.3, 0.7]  # Weights for Gaussian mixture model
MIXTURE_MEANS = [-1.5, 2.5]  # Means for Gaussian mixture model
MIXTURE_COVS = [0.1, 0.2]  # Covariances for Gaussian mixture model

# Constants for general testing
N_STARTS_FEW = 5  # Number of starts for tests that dont require convergence
N_STARTS_SOME = 10  # Number of starts for tests that converge reliably
N_SAMPLE_FEW = 100  # Number of samples for tests that dont require convergence
N_SAMPLE_SOME = 1000  # Number of samples for tests that converge reliably
N_SAMPLE_MANY = 5000  # Number of samples for tests that require convergence
STATISTIC_TOL = 0.2  # Tolerance when comparing distributions
N_CHAINS = 3  # Number of chains for ParallelTempering


def gaussian_llh(x):
    """Log-likelihood for Gaussian."""
    return float(norm.logpdf(x, loc=MU, scale=SIGMA).item())


def gaussian_nllh_grad(x):
    """Negative log-likelihood gradient for Gaussian."""
    return (x - MU) / (SIGMA**2)


def gaussian_nllh_hess(x):
    """Negative log-likelihood Hessian for Gaussian."""
    return np.array([(1 / (SIGMA**2))])


def gaussian_problem():
    """Defines a simple Gaussian problem."""

    def nllh(x):
        return -gaussian_llh(x)

    objective = pypesto.Objective(fun=nllh, grad=gaussian_nllh_grad)
    problem = pypesto.Problem(
        objective=objective, lb=LB_GAUSSIAN, ub=UB_GAUSSIAN
    )
    return problem


def gaussian_mixture_llh(x):
    """Log-likelihood for Gaussian mixture model."""
    return np.log(
        MIXTURE_WEIGHTS[0]
        * multivariate_normal.pdf(
            x, mean=MIXTURE_MEANS[0], cov=MIXTURE_COVS[0]
        )
        + MIXTURE_WEIGHTS[1]
        * multivariate_normal.pdf(
            x, mean=MIXTURE_MEANS[1], cov=MIXTURE_COVS[1]
        )
    )


def gaussian_mixture_problem():
    """Problem based on a mixture of Gaussians."""

    def nllh(x):
        return -gaussian_mixture_llh(x)

    objective = pypesto.Objective(fun=nllh)
    problem = pypesto.Problem(
        objective=objective, lb=LB_GAUSSIAN, ub=UB_GAUSSIAN, x_names=X_NAMES
    )
    return problem


def rosenbrock_problem():
    """Problem based on Rosenbrock objective."""
    objective = pypesto.Objective(fun=so.rosen, grad=so.rosen_der)

    dim_full = 2
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))

    problem = pypesto.Problem(
        objective=objective,
        lb=lb,
        ub=ub,
        x_fixed_indices=[1],
        x_fixed_vals=[2],
    )
    return problem


def create_petab_problem():
    """Creates a petab problem."""
    import os

    import petab.v1 as petab

    import pypesto.petab

    current_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(
        os.path.join(current_path, "..", "..", "doc", "example")
    )

    petab_problem = petab.Problem.from_yaml(
        dir_path + "/conversion_reaction/conversion_reaction.yaml"
    )

    importer = pypesto.petab.PetabImporter(petab_problem)
    problem = importer.create_problem()

    return problem


def prior(x):
    """Calculates the prior."""
    return multivariate_normal.pdf(x, mean=-1.0, cov=0.7)


def likelihood(x):
    """Calculates the likelihood."""
    return uniform.pdf(x, loc=-10.0, scale=20.0)[0]


def negative_log_posterior(x):
    """Calculates the negative log posterior."""
    return -np.log(likelihood(x)) - np.log(prior(x))


def negative_log_prior(x):
    """Calculates the negative log prior."""
    return -np.log(prior(x))
