#!/usr/bin/env python

from functools import partial
from typing import Optional, Union

import numpy as np

from pypesto.objective import (
    AggregatedObjective,
    NegLogParameterPriors,
    Objective,
)
from pypesto.problem import Problem

# M1
N = 10
mu1 = 0  # prior: mu1 ~ N(0, sigma1^2)
sigma1 = 1

# M2
N2_1 = 3
mu2_1 = -2  # prior: mu2_1 ~ N(-2, sigma2_1^2)
sigma2 = 2
N2_2 = N - N2_1
mu2_2 = 2  # prior: mu2_2 ~ N(2, sigma2_2^2)

# sigma is fixed to the true value
np.random.seed(0)
Y1 = np.random.normal(loc=mu1, scale=sigma1, size=N)
Y2_1 = np.random.normal(loc=mu2_1, scale=sigma2, size=N2_1)
Y2_2 = np.random.normal(loc=mu2_2, scale=sigma2, size=N2_2)
Y2 = np.concatenate([Y2_1, Y2_2])

# we choose the true model to be M2
Y, sigma = Y2, sigma2
true_params = np.array([mu2_1, mu2_2])


# evidence
def log_evidence_m1(data: np.ndarray, std: float):
    n = data.size
    y_sum = np.sum(data)
    y_sq_sum = np.sum(data**2)

    term1 = 1 / (np.sqrt(2 * np.pi) * std)
    log_term2 = -0.5 * np.log(n + 1)
    inside_exp = -0.5 / (std**2) * (y_sq_sum - (y_sum**2) / (n + 1))
    return n * np.log(term1) + log_term2 + inside_exp


def log_evidence_m2(data: np.ndarray, std: float):
    y1 = data[:N2_1]
    y2 = data[N2_1:]
    n = N2_1 + N2_2

    y_mean_1 = np.mean(y1)
    y_mean_2 = np.mean(y2)
    y_sq_sum = np.sum(y1**2) + np.sum(y2**2)

    term1 = (1 / (np.sqrt(2 * np.pi) * std)) ** n
    term2 = 1 / (np.sqrt(N2_1 + 1) * np.sqrt(N2_2 + 1))

    inside_exp = (
        -1
        / (2 * std**2)
        * (
            y_sq_sum
            + 8
            - (N2_1 * y_mean_1 - 2) ** 2 / (N2_1 + 1)
            - (N2_2 * y_mean_2 + 2) ** 2 / (N2_2 + 1)
        )
    )

    return np.log(term1) + np.log(term2) + inside_exp


true_log_evidence_m1 = log_evidence_m1(Y, sigma)
true_log_evidence_m2 = log_evidence_m2(Y, sigma)


# define likelihood for each model
def neg_log_likelihood(params: Union[np.ndarray, list], data: np.ndarray):
    # normal distribution
    mu, std = params
    n = data.size
    return (
        0.5 * n * np.log(2 * np.pi)
        + n * np.log(std)
        + np.sum((data - mu) ** 2) / (2 * std**2)
    )


def neg_log_likelihood_grad(params: Union[np.ndarray, list], data: np.ndarray):
    mu, std = params
    n = data.size
    grad_mu = -np.sum(data - mu) / (std**2)
    grad_std = n / std - np.sum((data - mu) ** 2) / (std**3)
    return np.array([grad_mu, grad_std])


def neg_log_likelihood_hess(params: Union[np.ndarray, list], data: np.ndarray):
    mu, std = params
    n = data.size
    hess_mu_mu = n / (std**2)
    hess_mu_std = 2 * np.sum(data - mu) / (std**3)
    hess_std_std = -n / (std**2) + 3 * np.sum((data - mu) ** 2) / (std**4)
    return np.array([[hess_mu_mu, hess_mu_std], [hess_mu_std, hess_std_std]])


def neg_log_likelihood_m2(
    params: Union[np.ndarray, list], data: np.ndarray, n1: int
):
    # normal distribution
    y1 = data[:n1]
    y2 = data[n1:]
    m1, m2, std = params

    neg_log_likelihood([m1, std], y1)
    term1 = neg_log_likelihood([m1, std], y1)
    term2 = neg_log_likelihood([m2, std], y2)
    return term1 + term2


def neg_log_likelihood_m2_grad(params: np.ndarray, data: np.ndarray, n1: int):
    m1, m2, std = params
    y1 = data[:n1]
    y2 = data[n1:]

    grad_m1, grad_std1 = neg_log_likelihood_grad([m1, std], y1)
    grad_m2, grad_std2 = neg_log_likelihood_grad([m2, std], y2)
    return np.array([grad_m1, grad_m2, grad_std1 + grad_std2])


def neg_log_likelihood_m2_hess(params: np.ndarray, data: np.ndarray, n1: int):
    m1, m2, std = params
    y1 = data[:n1]
    y2 = data[n1:]

    [[hess_m1_m1, hess_m1_std], [_, hess_std_std1]] = neg_log_likelihood_hess(
        [m1, std], y1
    )
    [[hess_m2_m2, hess_m2_std], [_, hess_std_std2]] = neg_log_likelihood_hess(
        [m2, std], y2
    )
    hess_m1_m2 = 0

    return np.array(
        [
            [hess_m1_m1, hess_m1_m2, hess_m1_std],
            [hess_m1_m2, hess_m2_m2, hess_m2_std],
            [hess_m1_std, hess_m2_std, hess_std_std1 + hess_std_std2],
        ]
    )


nllh_m1 = Objective(
    fun=partial(neg_log_likelihood, data=Y),
    grad=partial(neg_log_likelihood_grad, data=Y),
    hess=partial(neg_log_likelihood_hess, data=Y),
)
nllh_m2 = Objective(
    fun=partial(neg_log_likelihood_m2, data=Y, n1=N2_1),
    grad=partial(neg_log_likelihood_m2_grad, data=Y, n1=N2_1),
    hess=partial(neg_log_likelihood_m2_hess, data=Y, n1=N2_1),
)


def log_normal_density(x: float, mu: float, std: float):
    return (
        -1 / 2 * np.log(2 * np.pi)
        - 1 / 2 * np.log(std**2)
        - (x - mu) ** 2 / (2 * std**2)
    )


def log_normal_density_grad(x: float, mu: float, std: float):
    return -(x - mu) / (std**2)


def log_normal_density_hess(x: float, mu: float, std: float):
    return -1 / (std**2)


prior_m1 = NegLogParameterPriors(
    [
        {
            "index": 0,
            "density_fun": partial(log_normal_density, mu=mu1, std=sigma1),
            "density_dx": partial(log_normal_density_grad, mu=mu1, std=sigma1),
            "density_ddx": partial(
                log_normal_density_hess, mu=mu1, std=sigma1
            ),
        }
    ]
)
prior_m2 = NegLogParameterPriors(
    [
        {
            "index": 0,
            "density_fun": partial(log_normal_density, mu=mu2_1, std=sigma2),
            "density_dx": partial(
                log_normal_density_grad, mu=mu2_1, std=sigma2
            ),
            "density_ddx": partial(
                log_normal_density_hess, mu=mu2_1, std=sigma2
            ),
        },
        {
            "index": 1,
            "density_fun": partial(log_normal_density, mu=mu2_2, std=sigma2),
            "density_dx": partial(
                log_normal_density_grad, mu=mu2_2, std=sigma2
            ),
            "density_ddx": partial(
                log_normal_density_hess, mu=mu2_2, std=sigma2
            ),
        },
    ]
)

objective_m1 = AggregatedObjective(objectives=[nllh_m1, prior_m1])
objective_m2 = AggregatedObjective(objectives=[nllh_m2, prior_m2])

problem1 = Problem(
    objective=objective_m1,
    lb=[-10, 0],
    ub=[10, 10],
    x_names=["mu", "sigma"],
    x_scales=["lin", "lin"],
    x_fixed_indices=[1],
    x_fixed_vals=[sigma],
    x_priors_defs=prior_m1,
)

problem2 = Problem(
    objective=objective_m2,
    lb=[-10, -10, 0],
    ub=[10, 10, 10],
    x_names=["mu1", "mu2", "sigma"],
    x_scales=["lin", "lin", "lin"],
    x_fixed_indices=[2],
    x_fixed_vals=[sigma],
    x_priors_defs=prior_m2,
)


def batch_simulator_m1(
    param_batch: np.ndarray, std: Optional[np.ndarray] = None
):
    m = param_batch.flatten()
    if std is None:
        std = np.ones_like(m)
    return np.random.normal(loc=m, scale=std, size=(N, m.size)).T


def batch_simulator_m2(
    param_batch: np.ndarray, std: Optional[np.ndarray] = None
):
    m1, m2 = param_batch[:, 0], param_batch[:, 1]
    if std is None:
        std = np.ones_like(m1)
    y1 = np.random.normal(loc=m1, scale=std, size=(N2_1, m1.size)).T
    y2 = np.random.normal(loc=m2, scale=std, size=(N2_2, m2.size)).T
    return np.concatenate((y1, y2), axis=1)
