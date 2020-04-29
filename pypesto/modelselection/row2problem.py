import petab
from colorama import Fore
from typing import Dict, Union
import numpy as np
from ..petab import PetabImporter
from ..objective import Objective
from ..problem import Problem

from petab.C import PARAMETER_ID


def row2problem(petab_problem: Union[petab.problem, str],
                row: Dict[str, float],
                obj: Objective = None) -> Problem:
    """
    Create a pypesto.problem from a single, unambiguous model selection row
    and a petab.problem
    """
    # overwrite petab_problem by problem in case it refers to yaml
    if isinstance(petab_problem, str):
        petab_problem = petab.load_yaml(petab_problem)
    importer = PetabImporter(petab_problem)
    # chose standard objective in case none is provided
    if Objective is None:
        obj = importer.create_objective()
    pypesto_problem = importer.create_problem(obj)

    for par_id, par_val in row.items():
        if par_id not in petab_problem.x_ids:
            print(Fore.YELLOW + f'Warning: parameter {par_id} is not defined '
                                f'in PETab model. It will be ignored.')
            continue
        if not np.isnan(par_val):
            petab_problem.parameter_df.estimate.loc[par_id] = 0
            petab_problem.parameter_df.nominalValue.loc[par_id] = par_val
            # petab_problem.parameter_df.lowerBound.loc[par_id] = float("NaN")
            # petab_problem.parameter_df.upperBound.loc[par_id] = float("NaN")
        else:
            petab_problem.parameter_df.estimate.loc[par_id]= 1
            # petab_problem.parameter_df.nominalValue.loc[par_id] = float(
            # "NaN")

    importer = PetabImporter(petab_problem)
    if Objective is None:
        obj = importer.create_objective()
    pypesto_problem = importer.create_problem(obj)

    return pypesto_problem

