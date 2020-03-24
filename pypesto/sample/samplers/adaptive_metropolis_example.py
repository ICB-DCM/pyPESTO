import numpy as np
import math

import adaptive_metropolis

def simulate_observable(theta, t):
    return np.sum(theta) * np.power(t,2)/4

def l_p_c(theta, measurements, measurement_timepoints, sigma):
    measured_theta = 1
    llh = 0
    for i, t in enumerate(measurement_timepoints):
        simulation = simulate_observable(theta, measurement_timepoints)
        llh = llh - 0.5 * (math.log(2*math.pi*sigma) + math.pow((measurements[i] - simulation[i]), 2) / sigma)
    return llh

# initial sample/initial parameter values
theta = np.array([0, 0, 0])
theta_bounds_lower = np.array([-5, -5, -5])
theta_bounds_upper = np.array([10, 10, 10])

# measurements used in likelihood function
observable_measurements = np.array([1, 4, 9, 16, 25])
# time points for each measurements... not really necessary
observable_timepoints = np.array([0, 1, 2, 4, 5])
# noise of measurements
sigma = 0.5

# covariance matrix of sample/parameters
Sigma = np.identity(theta.size)

#Sigma = np.array([
#    [2, -1, 0],
#    [-1, 2, -1],
#    [0, -1, 2]
#])

options = {
    'debug': True,
    'Sigma': Sigma,
    'theta_bounds_lower': theta_bounds_lower,
    'theta_bounds_upper': theta_bounds_upper,
    'iterations': 10,
    'decay_rate': 0.5,
    'threshold_iteration': 5,
    'regularization_factor': 1e-6
}

result = adaptive_metropolis.adaptive_metropolis(lambda t: l_p_c(t, observable_measurements, observable_timepoints, sigma), theta, options)

from pprint import pprint
pprint(result)

