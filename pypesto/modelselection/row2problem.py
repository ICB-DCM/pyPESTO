import petab
from colorama import Fore
from typing import Dict
from numpy import isnan
from ..petab import PetabImporter
from ..objective import Objective
from ..problem import Problem

from petab.C import PARAMETER_ID

def row2problem(petab_problem: petab.problem, row: Dict[str, float],
                obj: Objective = None) -> Problem:
    importer = PetabImporter(petab_problem)
    if Objective is None:
        obj = importer.create_objective()
    pypesto_problem = importer.create_problem(obj)

    x_fixed = []
    x_free = []
    x_values = []
    x_names = pypesto_problem.x_names
    for par_id, par_val in row.items():
        if par_id not in petab_problem.get_model_parameters():
            print(Fore.YELLOW + f'Warning: parameter {par_id} is not defined '
                                f'in SBML model. It will be ignored.')
            continue
        if par_id not in pypesto_problem.x_names:
            print(Fore.YELLOW + f'Warning: parameter {par_id} was not found '
                                f'in pyPESTO.problem. It will be ignored.')
            continue
        if not isnan(par_val):
            x_fixed.append(x_names.index(par_id))
            x_values.append(par_val)
        else:
            x_free.append(x_names.index(par_id))
    # pypesto_problem.x_fixed_vals = x_values
    # pypesto_problem.x_fixed_indices = x_fixed
    pypesto_problem.fix_parameters(x_fixed, x_values)
    pypesto_problem.x_free_indices = x_free
