# Discrete Gillespie SIR disease dynamics model

module SIR

# Install dependencies
import Pkg
Pkg.add(["Catalyst", "OrdinaryDiffEq", "Zygote", "SciMLSensitivity"])

# Define reaction network
using Catalyst: @reaction_network
sir_model = @reaction_network begin
    r1, S + I --> 2I
    r2, I --> R
end r1 r2

# Ground truth parameter
p_true = [0.0001, 0.01]
# Initial state
u0 = [999; 1; 0]
# Time span
tspan = (0.0, 250.0)

# Formulate as ODE problem
using OrdinaryDiffEq: ODEProblem, solve, Tsit5
prob = ODEProblem(sir_model, u0, tspan, p_true)

# True trajectory
sol_true = solve(prob, Tsit5(), saveat=25)

# Observed data
using Random: randn, MersenneTwister
sigma = 40.0
rng = MersenneTwister(1234)
data = sol_true + sigma * randn(rng, size(sol_true))

using SciMLSensitivity: remake

# Define log-likelihood
function fun(p)
    # untransform parameters
    p = 10.0 .^ p
    # simulate
    _prob = remake(prob, p=p)
    sol_sim = solve(_prob, Tsit5(), saveat=25)
    # calculate log-likelihood
    0.5 * (log(2 * pi * sigma^2) + sum((sol_sim .- data).^2) / sigma^2)
end

# Define gradient and Hessian
using Zygote: gradient, hessian

function grad(p)
    gradient(fun, p)[1]
end

function hess(p)
    hessian(fun, p)[1]
end

end  # module
