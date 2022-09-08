# Simple linear regression test problem

module LR

using Random: randn, MersenneTwister
using ForwardDiff: gradient

rng = MersenneTwister(0)

# Number of parameters and data points
n_p = 8
n_y = 20
# True parameters
p_true = randn(rng, n_p)
# Regressors
X = randn(n_y, n_p)
# Observed data
eps = 0.1
y_obs = X * p_true + eps * randn(rng, n_y)

"""Least squares objective function."""
function fun(p)
    (y_obs - X * p)' * (y_obs - X * p) / eps^2
end

"""Gradient of least squares objective function."""
function grad(p)
    gradient(fun, p)
end

# Optimal parameters
p_opt = inv(X' * X) * X' * y_obs

end  # module
