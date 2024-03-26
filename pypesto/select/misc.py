"""Miscellaneous methods."""

import logging
from typing import Iterable

import pandas as pd
import petab
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
    x_guesses: Iterable[dict[str, float]] = None,
    hierarchical: bool = False,
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
    hierarchical:
        Whether the problem involves hierarchical optimization.

    Returns
    -------
    The pyPESTO select problem.
    """
    petab_problem = petab_select.ui.model_to_petab(model=model)[PETAB_PROBLEM]

    corrected_x_guesses = None
    if x_guesses is not None:
        corrected_x_guesses = correct_x_guesses(
            x_guesses=x_guesses,
            model=model,
            petab_problem=petab_problem,
            hierarchical=hierarchical,
        )

    importer = PetabImporter(
        petab_problem,
        hierarchical=hierarchical,
    )
    if objective is None:
        amici_model = importer.create_model(
            non_estimated_parameters_as_constants=False,
        )
        objective = importer.create_objective(
            model=amici_model,
        )
    pypesto_problem = importer.create_problem(
        objective=objective,
        x_guesses=corrected_x_guesses,
    )
    return pypesto_problem


def model_to_hierarchical_pypesto_problem(*args, **kwargs) -> Problem:
    """Create a hierarchical pyPESTO problem from a PEtab Select model.

    See :func:`model_to_pypesto_problem`.
    """
    pypesto_problem = model_to_pypesto_problem(
        *args,
        **kwargs,
        hierarchical=True,
    )
    return pypesto_problem


def correct_x_guesses(
    x_guesses: Iterable[dict[str, float]],
    model: Model,
    petab_problem: petab.Problem = None,
    hierarchical: bool = False,
):
    """Fix startpoint guesses passed between models of different sizes.

    Any parameter values in `x_guess` for parameters that are not estimated
    should be corrected by replacing them with
    - their corresponding values in `row` if possible, else
    - their corresponding nominal values in the `petab_problem.parameter_df`.

    Parameters
    ----------
    x_guesses:
        The startpoints to correct.
    model:
        The startpoints will be corrected to match this model.
    petab_problem:
        The model's corresponding PEtab problem. If this is not provided,
        it will be created from the `model`.
    hierarchical:
        Whether hierarchical optimization is used.

    Returns
    -------
    The corrected startpoint guesses.
    """
    # TODO reconsider whether correcting is a good idea (`x_guess` is no longer
    # the latest MLE then). Similar todo exists in
    # `ModelSelectorMethod.new_model_problem`.
    # TODO move to PEtab Select?
    # TODO may be issues, e.g. differences in bounds of parameters between
    #      different model PEtab problems is not accounted for, or if there are
    #      different parameters in the old/new PEtab problem.

    if petab_problem is None:
        petab_problem = petab_select.ui.model_to_petab(model=model)[
            PETAB_PROBLEM
        ]

    corrected_x_guesses = None
    if x_guesses is not None:
        corrected_x_guesses = []
        for x_guess in x_guesses:
            corrected_x_guess = []
            for parameter_id in petab_problem.parameter_df.index:
                if hierarchical:
                    if not pd.isna(
                        petab_problem.parameter_df.loc[
                            parameter_id, "parameterType"
                        ]
                    ):
                        continue

                # Use the `x_guess` value, if the parameter is to be estimated.
                if (
                    petab_problem.parameter_df[ESTIMATE].loc[parameter_id] == 1
                    and parameter_id in x_guess
                ):
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
    return corrected_x_guesses
