import petab
from colorama import Fore
from typing import Union
import numpy as np
from ..petab import PetabImporter
from ..objective import Objective
from ..problem import Problem
import pandas as pd

from petab.C import NOMINAL_VALUE, ESTIMATE

YAML_FILENAME_COLUMN = "SBML"
MODEL_NAME_COLUMN = "ModelId"


def row2problem(row: dict,
                petab_problem: Union[petab.Problem, str] = None,
                obj: Objective = None) -> Problem:
    """
    Create a pypesto.Problem from a single, unambiguous model selection row.
    Optional petab.Problem and objective function can be provided to overwrite
    model selection yaml entry and default PetabImporter objective
    respectively.
    
    Parameters
    ----------
    row:
        A single, unambiguous model selection row.
    petab_problem:
        The petab problem for which to perform model selection.
    obj:
        The objective to modify for model selection.
    
    Returns
    -------
    problem:
        The problem containing correctly fixed parameter values.
    """
    # overwrite petab_problem by problem in case it refers to yaml
    if petab_problem is None:
        petab_problem = row[YAML_FILENAME_COLUMN].str
    if isinstance(petab_problem, str):
        petab_problem = petab.Problem.from_yaml(petab_problem)
    importer = PetabImporter(petab_problem)

    # drop row entries not referring to parameters
    for key in [YAML_FILENAME_COLUMN, MODEL_NAME_COLUMN]:
        row.pop(key)

    for par_id, par_val in row.items():
        if par_id not in petab_problem.x_ids:
            print(Fore.YELLOW + f'Warning: parameter {par_id} is not defined '
                                f'in PETab model. It will be ignored.')
            continue
        if not np.isnan(par_val):
            petab_problem.parameter_df[ESTIMATE].loc[par_id] = 0
            petab_problem.parameter_df[NOMINAL_VALUE].loc[par_id] = par_val
            # petab_problem.parameter_df.lowerBound.loc[par_id] = float("NaN")
            # petab_problem.parameter_df.upperBound.loc[par_id] = float("NaN")
        else:
            petab_problem.parameter_df[ESTIMATE].loc[par_id] = 1
            # petab_problem.parameter_df.nominalValue.loc[par_id] = float(
            # "NaN")

    # chose standard objective in case none is provided
    importer = PetabImporter(petab_problem)
    if Objective is None:
        obj = importer.create_objective()
    pypesto_problem = importer.create_problem(obj)

    return pypesto_problem
