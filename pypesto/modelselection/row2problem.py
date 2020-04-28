import petab
from colorama import Fore
from typing import Dict
from numpy import isnan
from ..petab import PetabImporter
from ..objective import Objective
from ..problem import Problem

def row2problem(petab_problem: petab.problem, row: Dict[str, float],
                obj: Objective = None) -> Problem:
    importer = PetabImporter(petab_problem)
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
            petab_problem.parameter_df.nominalValue.loc[par_id] = float("NaN")
            
    importer = PetabImporter(petab_problem)
    if Objective is None:
        obj = importer.create_objective()
    pypesto_problem = importer.create_problem(obj)
