from colorama import Fore
import numpy as np
from typing import Dict, Iterable, Sequence, Set, Union

import petab
from petab.C import NOMINAL_VALUE, ESTIMATE
from petab_select import (
    Model,
)

from ..objective import Objective
from ..petab import PetabImporter
from ..problem import Problem

import logging
logger = logging.getLogger(__name__)


# FIXME rename to e.g. `model2problem`
def row2problem(
    model: Model,
    petab_problem: Union[petab.Problem, str] = None,
    obj: Objective = None,
    x_guess: Sequence[float] = None,
) -> Problem:
    """
    Create a pypesto.Problem from a single, unambiguous model selection row.
    Optional petab.Problem and objective function can be provided to overwrite
    model selection yaml entry and default PetabImporter objective
    respectively.

    Parameters
    ----------
    petab_problem:
        The petab problem for which to perform model selection.

    obj:
        The objective to modify for model selection.

    x_guess:
        A startpoint to be used in the multistart optimization. For example,
        this could be the maximum likelihood estimate from another model.
        Values in `x_guess` for parameters that are not estimated will be
        ignored.

    Returns
    -------
    problem:
        The problem containing correctly fixed parameter values.
    """
    # # overwrite petab_problem by problem in case it refers to yaml
    # # TODO if yaml is specified in the model spec file, then a new problem
    # # might be created for each model row. This may be undesirable as the
    # # same model may be compiled for each model row with the same YAML value
    # if petab_problem is None and YAML_FILENAME in row.keys():
    #     raise NotImplementedError()
    #     # TODO untested
    #     # YAML_FILENAME_COLUMN is not currently specified in the model
    #     # specifications file (instead, the SBML .xml file is)
    #     petab_problem = row[YAML_FILENAME]
    # if isinstance(petab_problem, str):
    #     petab_problem = petab.Problem.from_yaml(petab_problem)

    #  # drop row entries not referring to parameters
    #  # TODO switch to just YAML_FILENAME
    #  for key in [YAML_FILENAME, SBML_FILENAME, MODEL_ID]:
    #      if key in row.keys():
    #          row.pop(key)
    # row_parameters = {k: row[k] for k in row if k not in NOT_PARAMETERS}

    for par_id, par_val in model.parameters.items():
        if par_id not in petab_problem.x_ids:
            logger.info('%sWarning: parameter %s is not defined '
                        'in PETab model. It will be ignored.',
                        Fore.YELLOW,
                        par_id)
            continue
        if not np.isnan(par_val):
            petab_problem.parameter_df[ESTIMATE].loc[par_id] = 0
            petab_problem.parameter_df[NOMINAL_VALUE].loc[par_id] = par_val
        else:
            petab_problem.parameter_df[ESTIMATE].loc[par_id] = 1

    # Any parameter values in `x_guess` for parameters that are not estimated
    # should be filtered out and replaced with
    # - their corresponding values in `row` if possible, else
    # - their corresponding nominal values in the `petab_problem.parameter_df`.
    # TODO reconsider whether filtering is a good idea (x_guess is no longer
    # the latest MLE then). Similar todo exists in
    # `ModelSelectorMethod.new_model_problem`.
    if x_guess is not None:
        filtered_x_guess = []
        for par_id, par_val in x_guess:
            if petab_problem.parameter_df[ESTIMATE].loc[par_id] == 1:
                filtered_x_guess.append(par_val)
            else:
                if par_id in model.parameters:
                    filtered_x_guess.append(model.parameters[par_id])
                else:
                    filtered_x_guess.append(
                        petab_problem.parameter_df[NOMINAL_VALUE].loc[par_id])
        x_guesses = [filtered_x_guess]
    else:
        x_guesses = None

    # chose standard objective in case none is provided
    importer = PetabImporter(petab_problem)
    if obj is None:
        obj = importer.create_objective()
    pypesto_problem = importer.create_problem(obj, x_guesses=x_guesses)

    return pypesto_problem


def get_x_fixed_estimated(
        x_ids: Set[str],
        model_generator: Iterable[Dict[str, Union[str, float]]]) -> Set[str]:
    '''
    Get parameters that are fixed in at least one model, but should be
    considered estimated. This may not be a feature worth supporting.
    TODO Consider instead a single float, or `-`, as the two valid parameter
    specification symbols, where only `-` represents an estimated parameter,
    don't support specification of multiple possible float values.
    '''
    pass
