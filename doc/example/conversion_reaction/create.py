from petab.C import *
import petab

import pandas as pd
import numpy as np

a0 = 1
b0 = 0
k1 = 0.8
k2 = 0.6


def analytical_a(t, a0=a0, b0=b0, k1=k1, k2=k2):
    return k2 * (a0 + b0) / (k1 + k2) \
           + (a0 - k2 * (a0 + b0) / (k1 + k2)) * np.exp(-(k1 + k2) * t)


# problem --------------------------------------------------------------------

condition_df = pd.DataFrame(data={
    CONDITION_ID: ['c0'],
}).set_index([CONDITION_ID])

times = np.linspace(0, 3, 10)
nt = len(times)
simulations = [analytical_a(t, 1, 0, 0.8, 0.6)
               for t in times]
sigma = 0.02
measurements = simulations + sigma * np.random.randn(nt)

measurement_df = pd.DataFrame(data={
    OBSERVABLE_ID: ['obs_a'] * nt,
    SIMULATION_CONDITION_ID: ['c0'] * nt,
    TIME: times,
    MEASUREMENT: measurements
})

observable_df = pd.DataFrame(data={
    OBSERVABLE_ID: ['obs_a'],
    OBSERVABLE_FORMULA: ['A'],
    NOISE_FORMULA: [sigma]
}).set_index([OBSERVABLE_ID])

parameter_df = pd.DataFrame(data={
    PARAMETER_ID: ['k1', 'k2'],
    PARAMETER_SCALE: [LOG] * 2,
    LOWER_BOUND: [1e-5] * 2,
    UPPER_BOUND: [1e5] * 2,
    NOMINAL_VALUE: [k1, k2],
    ESTIMATE: [1, 1],
}).set_index(PARAMETER_ID)


petab.write_condition_df(condition_df, "conditions.tsv")
petab.write_measurement_df(measurement_df, "measurements.tsv")
petab.write_observable_df(observable_df, "observables.tsv")
petab.write_parameter_df(parameter_df, "parameters.tsv")

yaml_config = {
    FORMAT_VERSION: 1,
    PARAMETER_FILE: "parameters.tsv",
    PROBLEMS: [{
        SBML_FILES: ["model_conversion_reaction.xml"],
        CONDITION_FILES: ["conditions.tsv"],
        MEASUREMENT_FILES: ["measurements.tsv"],
        OBSERVABLE_FILES: ["observables.tsv"]
    }]
}
petab.write_yaml(yaml_config, "conversion_reaction.yaml")

# validate written PEtab files
problem = petab.Problem.from_yaml("conversion_reaction.yaml")
petab.lint_problem(problem)
