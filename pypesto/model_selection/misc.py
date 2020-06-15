from colorama import Fore
import itertools
import math
import numpy as np
import tempfile
from typing import Dict, Iterable, List, Sequence, Set, Union

import petab
from petab.C import NOMINAL_VALUE, ESTIMATE

from ..objective import Objective
from ..petab import PetabImporter
from ..problem import Problem

from .constants import (
    ESTIMATE_SYMBOL_INTERNAL,
    ESTIMATE_SYMBOL_UI,
    HEADER_ROW,
    MODEL_ID_COLUMN,
    NOT_PARAMETERS,
    PARAMETER_DEFINITIONS_START,
    PARAMETER_VALUE_DELIMITER,
    YAML_FILENAME,
    YAML_FILENAME_COLUMN,
)

import logging
logger = logging.getLogger(__name__)

def row2problem(row: dict,
                petab_problem: Union[petab.Problem, str] = None,
                obj: Objective = None,
                x_guess: Sequence[float] = None) -> Problem:
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
    # overwrite petab_problem by problem in case it refers to yaml
    # TODO if yaml is specified in the model spec file, then a new problem
    # might be created for each model row. This may be undesirable as the same
    # model might be compiled for each model row with the same YAML value
    if petab_problem is None and YAML_FILENAME in row.keys():
        raise NotImplementedError()
        # TODO untested
        # YAML_FILENAME_COLUMN is not currently specified in the model
        # specifications file (instead, the SBML .xml file is)
        petab_problem = row[YAML_FILENAME]
    if isinstance(petab_problem, str):
        petab_problem = petab.Problem.from_yaml(petab_problem)

    ## drop row entries not referring to parameters
    ## TODO switch to just YAML_FILENAME
    #for key in [YAML_FILENAME, SBML_FILENAME, MODEL_ID]:
    #    if key in row.keys():
    #        row.pop(key)
    row_parameters = {k: row[k] for k in row if k not in NOT_PARAMETERS}

    for par_id, par_val in row_parameters.items():
    #for par_id, par_val in row.items():
        if par_id not in petab_problem.x_ids:
            logger.info(Fore.YELLOW + f'Warning: parameter {par_id} is not defined '
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
                if par_id in row_parameters:
                    filtered_x_guess.append(row_parameters[par_id])
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


def aic(n_estimated: int, nllh: float) -> float:
    """
    Calculate the Akaike information criterion for a model.

    Arguments
    ---------
    n_estimated:
        The number of estimated parameters in the model.

    nllh:
        The negative log likelihood (e.g. the `optimize_result.list[0].fval`
        attribute of the return value of `pypesto.minimize`).
    """
    return 2*(n_estimated + nllh)


def bic(n_estimated: int, n_measurements: int, nllh: float):
    """
    Calculate the Bayesian information criterion for a model.
    TODO untested

    Arguments
    ---------
    n_estimated:
        The number of estimated parameters in the model.

    n_observations:
        The number of measurements used in the objective function of the model
        (e.g. `len(petab_problem.measurement_df)`).

    nllh:
        The negative log likelihood (e.g. the `optimize_result.list[0].fval`
        attribute of the return value of `pypesto.minimize`).
    """
    return n_estimated*math.log(n_measurements) + 2*nllh


def _replace_estimate_symbol(parameter_definition: List[str]) -> List:
    """
    Converts the user-friendly symbol for estimated parameters, to the internal
    symbol.

    Arguments
    ---------
    parameter_definition:
        A definition for a single parameter from a row of the model
        specification file. The definition should be split into a list by
        PARAMETER_VALUE_DELIMITER.

    Returns
    -------
    The parameter definition, with the user-friendly estimate symbol
    substituted for the internal symbol.
    """
    return [ESTIMATE_SYMBOL_INTERNAL if p == ESTIMATE_SYMBOL_UI else p
            for p in parameter_definition]


def unpack_file(file_name: str):
    """
    Unpacks a model specification file into a new temporary file, which is
    returned.

    Arguments
    ---------
    file_name:
        The name of the file to be unpacked.

    Returns
    -------
    A temporary file object, which is the unpacked file.

    TODO
        - Consider alternatives to `_{n}` suffix for model `modelId`
        - How should the selected model be reported to the user? Remove the
          `_{n}` suffix and report the original `modelId` alongside the
          selected parameters? Generate a set of PEtab files with the
          chosen SBML file and the parameters specified in a parameter or
          condition file?
        - Don't "unpack" file if it is already in the unpacked format
        - Sort file after unpacking
        - Remove duplicates?
    """
    expanded_models_file = tempfile.NamedTemporaryFile(mode='r+',
                                                       delete=False)
    with open(file_name) as fh:
        with open(expanded_models_file.name, 'w') as ms_f:
            # could replace `else` condition with ms_f.readline() here, and
            # remove `if` statement completely
            for line_index, line in enumerate(fh):
                # Skip empty/whitespace-only lines
                if not line.strip():
                    continue
                if line_index != HEADER_ROW:
                    columns = line2row(line, unpacked=False)
                    parameter_definitions = [
                        _replace_estimate_symbol(
                            definition.split(PARAMETER_VALUE_DELIMITER))
                        for definition in columns[
                            PARAMETER_DEFINITIONS_START:
                        ]
                    ]
                    for index, selection in enumerate(itertools.product(
                            *parameter_definitions
                    )):
                        # TODO change MODEL_ID_COLUMN and YAML_ID_COLUMN
                        # to just MODEL_ID and YAML_FILENAME?
                        ms_f.write(
                            '\t'.join([
                                columns[MODEL_ID_COLUMN]+f'_{index}',
                                columns[YAML_FILENAME_COLUMN],
                                *selection
                            ]) + '\n'
                        )
                else:
                    ms_f.write(line)
    return expanded_models_file


def line2row(line: str,
             delimiter: str = '\t',
             unpacked: bool = True,
             convert_parameters_to_float: bool = True) -> List:
    """
    Convert a line from the model specifications file, to a list of column
    values. No header information is returned.

    Arguments
    ---------
    line:
        A line from a file with delimiter-separated columns.

    delimiter:
        The string that separates columns in the file.

    unpacked:
        If False, parameter values are not converted to float.

    Returns
    -------
    A list of column values. Parameter values are converted to type `float`.
    """
    columns = line.strip().split(delimiter)
    metadata = columns[:PARAMETER_DEFINITIONS_START]
    if unpacked and convert_parameters_to_float:
        parameters = [float(p) for p in columns[PARAMETER_DEFINITIONS_START:]]
    else:
        parameters = columns[PARAMETER_DEFINITIONS_START:]
    return metadata + parameters


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
