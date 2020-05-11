import petab
from colorama import Fore
from typing import Union
import numpy as np
from ..petab import PetabImporter
from ..objective import Objective
from ..problem import Problem
from .constants import MODEL_NAME_COLUMN, YAML_FILENAME_COLUMN

from petab.C import NOMINAL_VALUE, ESTIMATE


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
    # TODO if yaml is specified in the model spec file, then a new problem
    # might be created for each model row. This may be undesirable as the same
    # model might be compiled for each model row with the same YAML value
    if petab_problem is None and YAML_FILENAME_COLUMN in row.keys():
        # TODO untested
        # YAML_FILENAME_COLUMN is not currently specified in the model
        # specifications file (instead, the SBML .xml file is)
        petab_problem = row[YAML_FILENAME_COLUMN]
    if isinstance(petab_problem, str):
        petab_problem = petab.Problem.from_yaml(petab_problem)
    importer = PetabImporter(petab_problem)

    # drop row entries not referring to parameters
    for key in [YAML_FILENAME_COLUMN, MODEL_NAME_COLUMN]:
        if key in row.keys():
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
    if obj is None:
        obj = importer.create_objective()
    pypesto_problem = importer.create_problem(obj)

    return pypesto_problem
