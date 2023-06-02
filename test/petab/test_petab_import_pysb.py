"""
This is for testing the petab import.
"""

import logging
import os

import numpy as np
import petab
import petabtests
import yaml

import pypesto.optimize as optimize
from pypesto.petab import PetabImporterPysb

# In CI, bionetgen is install here
BNGPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'BioNetGen-2.8.5')
)
if 'BNGPATH' not in os.environ:
    logging.warning(f"Env var BNGPATH was not set. Setting to {BNGPATH}")
    os.environ['BNGPATH'] = BNGPATH


def test_petab_pysb_optimization():
    test_case = '0001'
    test_case_dir = os.path.join(
        petabtests.CASES_DIR / 'v2.0.0' / 'pysb', test_case
    )
    petab_yaml = os.path.join(test_case_dir, f'_{test_case}.yaml')
    solution_yaml = os.path.join(test_case_dir, f'_{test_case}_solution.yaml')

    # expected results
    with open(solution_yaml) as f:
        solution = yaml.full_load(f)

    petab_problem = petab.Problem.from_yaml(petab_yaml)

    importer = PetabImporterPysb(petab_problem)
    problem = importer.create_problem()

    # ensure simulation result for true parameters matches
    assert np.isclose(
        problem.objective(petab_problem.x_nominal), -solution[petabtests.LLH]
    )

    optimizer = optimize.ScipyOptimizer()
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=10,
        progress_bar=False,
    )
    fvals = np.array(result.optimize_result.fval)

    # ensure objective after optimization is not worse than for true parameters
    assert np.all(fvals <= -solution[petabtests.LLH])
