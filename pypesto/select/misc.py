"""Miscellaneous methods."""
import logging
from typing import Dict, Iterable

import petab_select.ui
from petab.C import ESTIMATE, NOMINAL_VALUE
from petab_select import Model, parameter_string_to_value
from petab_select.constants import PETAB_PROBLEM

from ..objective import Objective
from ..petab import PetabImporter
from ..problem import Problem

logger = logging.getLogger(__name__)


def model_to_pypesto_problem(
    model: Model,
    objective: Objective = None,
    x_guesses: Iterable[Dict[str, float]] = None,
) -> Problem:
    """Create a pyPESTO problem from a PEtab Select model.

    Parameters
    ----------
    model:
        The model.
    objective:
        The pyPESTO objective.
    x_guesses:
        Startpoints to be used in the multi-start optimization. For example,
        this could be the maximum likelihood estimate from another model.
        Each dictionary has parameter IDs as keys, and parameter values as
        values.
        Values in `x_guess` for parameters that are not estimated will be
        ignored and replaced with their value from the PEtab Select model, if
        defined, else their nominal value in the PEtab parameters table.

    Returns
    -------
        The pyPESTO problem.
    """
    # Any parameter values in `x_guess` for parameters that are not estimated
    # should be corrected by replacing them with
    # - their corresponding values in `row` if possible, else
    # - their corresponding nominal values in the `petab_problem.parameter_df`.
    # TODO reconsider whether correcting is a good idea (`x_guess` is no longer
    # the latest MLE then). Similar todo exists in
    # `ModelSelectorMethod.new_model_problem`.

    petab_problem = petab_select.ui.model_to_petab(model=model)[PETAB_PROBLEM]

    # TODO may be issues, e.g. differences in bounds of parameters between
    #      different model PEtab problems is not accounted for, or if there are
    #      different parameters in the old/new PEtab problem.

    # TODO move to PEtab Select?
    corrected_x_guesses = None
    if x_guesses is not None:
        corrected_x_guesses = []
        for x_guess in x_guesses:
            corrected_x_guess = []
            for parameter_id in petab_problem.parameter_df.index:
                # Use the `x_guess` value, if the parameter is to be estimated.
                if petab_problem.parameter_df[ESTIMATE].loc[parameter_id] == 1:
                    corrected_value = x_guess[parameter_id]
                # Else use the PEtab Select model parameter value, if defined.
                elif parameter_id in model.parameters:
                    corrected_value = parameter_string_to_value(
                        model.parameters[parameter_id]
                    )
                # Default to the PEtab parameter table nominal value
                else:
                    corrected_value = petab_problem.parameter_df.loc[
                        parameter_id, NOMINAL_VALUE
                    ]
                corrected_x_guess.append(corrected_value)
            corrected_x_guesses.append(corrected_x_guess)

    importer = PetabImporter(petab_problem)
    if objective is None:
        objective = importer.create_objective()
    pypesto_problem = importer.create_problem(
        objective=objective,
        x_guesses=corrected_x_guesses,
    )
    return pypesto_problem
