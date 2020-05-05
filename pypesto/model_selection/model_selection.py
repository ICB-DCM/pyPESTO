import itertools
import tempfile
from typing import Dict, Iterator, List, Tuple

import petab
import numpy as np
import math
from ..problem import Problem
from ..optimize import minimize
from ..result import Result

from .tmp_row2problem import row2problem

MODEL_ID = "ModelId"
SBML_FILENAME = "SBML"
# Zero-indexed column indices
MODEL_NAME_COLUMN = 0
SBML_FILENAME_COLUMN = 1
PARAMETER_DEFINITIONS_START = 2
HEADER_ROW = 0
PARAMETER_VALUE_DELIMITER = ';'
ESTIMATE_SYMBOL_UI = '-'
# here, 'nan' is a string as it will be written to a (temporary) file. The
# actual internal symbol is float('nan')
ESTIMATE_SYMBOL_INTERNAL = 'nan'


class ModelSelectionProblem:
    """

    """
    def __init__(self,
                 petab_problem: petab.problem,
                 row: Dict[str, float],
                 valid: bool = True,
                 autorun: bool = True):
        """
        Arguments
        ---------
        petab_problem:
            A petab problem that includes the parameters defined in the model
            specification file.
        row:
            A single row from the model specification file, in the format that
            is returned by `ModelSelector.model_generator()`.
        valid:
            If `False`, the model will not be tested.
        autorun:
            If `False`, the model parameters will not be estimated. Allows
            users to manually call pypesto.minimize with custom options, then
            `set_result()`.
        TODO: constraints
        """
        self.petab_problem = petab_problem
        self.row = row
        self.valid = valid

        self.AIC = None
        self.BIC = None

        if self.valid:
            self.n_estimated = sum([1 if p == ESTIMATE_SYMBOL_UI else 0
                                    for header, p in row.items()
                                    if header not in [MODEL_ID,
                                                      SBML_FILENAME]])
            self.n_edata = None  # TODO for BIC
            self.model_id = row[MODEL_ID]
            self.SBML_filename = row[SBML_FILENAME]

            self.pypesto_problem = row2problem(row, petab_problem)

            self.minimize_result = None

            if autorun:
                self.set_result(minimize(self.pypesto_problem))

    def set_result(self, result: Result):
        self.minimize_result = result
        self.AIC = self.calculate_AIC()
        self.BIC = self.calculate_BIC()

    def calculate_AIC(self):
        # TODO: test, get objective fval for best parameter estimates from
        # minimize_result
        return 2*(self.n_estimated - math.log(self.minimize_result[0].fval))

    def calculate_BIC(self):
        # TODO: test, implement self.n_data in `__init__`
        # TODO: find out how to get size of experimental data
        #       petab.problem
        return self.n_estimated*math.log(self.n_data) - \
            2*math.log(self.minimize_result[0].fval)

    def calibrate(self):
        # TODO settings?
        # 100 starts, SingleCoreEngine, ScipyOptimizer
        result = minimize(self.pypesto_problem)
        self.set_result(result)


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
    """
    expanded_models_file = tempfile.NamedTemporaryFile(mode='r+',
                                                       delete=False)
    with open(file_name) as fh:
        with open(expanded_models_file.name, 'w') as ms_f:
            for line_index, line in enumerate(fh):
                if line_index != HEADER_ROW:
                    columns = line2row(line)
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
                        ms_f.write(
                            '\t'.join([
                                columns[MODEL_NAME_COLUMN]+f'_{index}',
                                columns[SBML_FILENAME_COLUMN],
                                *selection
                            ]) + '\n'
                        )
                else:
                    ms_f.write(line)
    return expanded_models_file


def line2row(line: str, delimiter: str = '\t') -> List:
    """
    Convert a line from the model specifications file, to a list of column
    values. No header information is returned.

    Arguments
    ---------
    line:
        A line from a file with delimiter-separated columns.

    delimiter:
        The string that separates columns in the file.

    Returns
    -------
    A list of column values.
    """
    return line.strip().split(delimiter)


class ModelSelector:
    def __init__(
            self,
            petab_problem: petab.problem,
            specification_filename: str
    ):
        self.petab_problem = petab_problem
        self.specification_file = unpack_file(specification_filename)
        self.header = line2row(self.specification_file.readline())
        self.parameter_ids = self.header[PARAMETER_DEFINITIONS_START:]

        self.selection_history = {}

        # Dictionary of method names as keys, with a dictionary as the values.
        # In the dictionary, keys will be modelId, criterion value
        # TODO
        self.results = {}

    def model_generator(self, index: int = None) -> Iterator[Dict[str, str]]:
        """
        Generates models from a model specification file.

        Argument
        --------
        index:
            If None, all models from the model specification file are yielded.
            If not None, only the model at line number `index` is returned.
        specification is returned.

        Returns
        -------
        Models, one model at a time, as a dictionary, where the keys are the
        column headers in the model specification file, and the values are the
        respective column values in a row of the model specification file.
        """
        # If index is specified, return the specific model
        # 1+index to skip the header row
        if index is not None:
            self.specification_file.seek(1+index)
            return dict(zip(self.header,
                            line2row(next(self.specification_file))))

        # Go to start of file, after header
        # self.specification_file.seek(1)
        # readline() in ModelSelector.__init__ skips header
        for line in self.specification_file:
            yield dict(zip(self.header, line2row(line)))

    def select(self, method: str, criterion: str):
        """
        Runs a model selection algorithm. The result is the selected model for
        the current run, independent of previous `select()` calls.

        Arguments
        ---------
        method:
            The model selection algorithm.

        criterion:
            The criterion used by `ModelSelectorMethod.compare()`, in which
            currently implemented criterion can be found.
        """
        if method == 'forward':
            selector = ForwardSelector(self.petab_problem,
                                       self.model_generator,
                                       criterion,
                                       self.parameter_ids,
                                       self.selection_history)
            result, self.selection_history = selector()
        elif method == 'backward':
            selector = ForwardSelector(self.petab_problem,
                                       self.model_generator,
                                       criterion,
                                       self.parameter_ids,
                                       self.selection_history,
                                       reverse=True)
            result, self.selection_history = selector()
        else:
            raise NotImplementedError(f'Model selection algorithm: {method}.')

        # TODO: Reconsider return value. `result` could be stored in attribute,
        # then no values need to be returned, and users can request values
        # manually.
        return result, self.selection_history


class ModelSelectorMethod:
    # the calling child class should have self.criterion defined
    def compare(self,
                old: ModelSelectionProblem,
                new: ModelSelectionProblem) -> bool:
        """
        Compares models by criterion.

        Arguments
        ---------
        old:
            A `ModelSelectionProblem` that has already been optimized.
        new:
            See `old`.

        Returns
        -------
        `True`, if `new` is superior to `old` by the criterion, else `False`.
        """
        if self.criterion == 'AIC':
            return new.get_AIC() < old.get_AIC()
        elif self.criterion == 'BIC':
            return new.get_BIC() < old.get_BIC()
        else:
            raise NotImplementedError('Model selection criterion: '
                                      f'{self.criterion}.')


class ForwardSelector(ModelSelectorMethod):
    """
    here it is assumed that that there is only one petab_problem
    """
    def __init__(self,
                 petab_problem: petab.problem,
                 model_generator: Iterator[Dict[str, str]],
                 criterion: str,
                 parameter_ids: List[str],
                 selection_history: Dict[str, Dict],
                 reverse: bool = False):
        self.petab_problem = petab_problem
        self.model_generator = model_generator
        self.criterion = criterion
        self.parameter_ids = parameter_ids
        self.selection_history = selection_history
        self.reverse = reverse

    def new_direction_problem(self) -> ModelSelectionProblem:
        """
        Produces a virtual initial model that can be used to identify
        models in `self.model_generator()` that are relatively more (or less,
        if `self.reverse` is `True` to indicate backward selection) complex
        compared to the initial model.

        Returns
        -------
        If `self.reverse` is `True`, returns a model with all parameters
        estimated (the initial model for backward selection), else returns a
        model with all parameters zero (the initial model for forward
        selection). Models are returned as a `ModelSelectionProblem`.
        The returned model will have `ModelSelectionProblem.valid=False`, to
        ensure that the model is not considered for selection.

        TODO:
            fail gracefully if no models are selected after the selection
            algorithm is run with this initial model, so this model is never
            reported as a possible model.
        """
        if self.reverse:
            # TODO ESTIMATE_SYMBOL_INTERNAL
            return ModelSelectionProblem(
                self.petab_problem,
                dict(zip(self.parameter_ids,
                         [float("NaN")]*len(self.parameter_ids),)),
                valid=False
            )
        else:
            return ModelSelectionProblem(
                self.petab_problem,
                dict(zip(self.parameter_ids, [0]*len(self.parameter_ids),)),
                valid=False
            )

    def __call__(self):
        """
        Runs the forward (or backward, if `self.reverse=True`) selection
        algorithm.

        Returns
        -------
        A tuple, where the first element is the selected model, as a
        `ModelSelectionProblem`, and the second element is a dictionary that
        describes the tested models, where the keys are model Ids, and the
        values are dictionaries with the keys 'AIC', 'BIC', and
        'compared_model_id'.
        """
        # self.setup_direction(self.direction)
        ms_problem = self.new_direction_problem()
        proceed = True

        while proceed:
            proceed = False
            model_candidate_indices = self.get_next_step_candidates(
                ms_problem.row)
            # Error if no valid candidate models are found. May occur if
            # all models have already been tested
            if not model_candidate_indices and not ms_problem.valid:
                raise Exception('No valid candidate models found.')
            for ind in model_candidate_indices:
                candidate_model = ModelSelectionProblem(
                    self.petab_problem,
                    self.model_generator(ind))

                self.selection_history[candidate_model.model_id] = {
                    'AIC': candidate_model.AIC,
                    'BIC': candidate_model.BIC,
                    'compared_model_id': (
                        'PYPESTO_INITIAL_MODEL' if not ms_problem.valid
                        else ms_problem.model_id
                    )
                }

                # The initial model from self.new_direction_problem() is only
                # for complexity comparison, and is not a real model.
                if not ms_problem.valid:
                    ms_problem = candidate_model
                    proceed = True
                    continue

                if self.compare(ms_problem, candidate_model):
                    ms_problem = candidate_model
                    proceed = True

        return ms_problem, self.selection_history

    def relative_complexity(self, old: float, new: float) -> int:
        """
        Currently, non-zero fixed values are considered equal in complexity to
        estimated values.
        """
        # if both parameters are equal "complexity", e.g. both are zero,
        # both are estimated, or both are non-zero values.
        if (old == new == 0) or \
           (math.isnan(old) and math.isnan(new)) or (
               not math.isnan(old) and
               not math.isnan(new) and
               old != 0 and
               new != 0
           ):
            return 0
        # return 1 if the new parameter is fixed or estimated, and the old
        # parameter is 0
        elif old == 0 and new != 0:
            return 1
        # return -1 if the new parameter is 0, and the old parameter is fixed
        # or estimated
        elif old != 0 and new == 0:
            return -1

    def get_next_step_candidates(self,
                                 old_model: Dict,
                                 strict=True) -> List[int]:
                                 #conf_dict: Dict[str, float],
                                 #direction=0
        """
        Identifies models are have minimal changes in complexity compared to
        `old_model`. Note: models that should be ignored are assigned a
        complexity of `float('nan')`.

        Parameters
        ----------
        old_model:
            The model that will be used to calculate the relative complexity of
            other models. Note: not a `ModelSelectionProblem`, but a dictionary
            in the format that is returned by
            `ModelSelector.model_generator()`.
        strict:
            If `True`, only models that strictly add (for forward selection) or
            remove (for backward selection) parameters compared to `old_model`
            will be returned.
            TODO: Is `strict=False` useful?

        Returns
        -------
        A list of indices from `self.model_generator()`, of models that are
        minimally more (for forward selection) or less (for backward selection)
        complex compared to the model described by `old_model`.
        """
        # the nth element in this list is the relative complexity for the nth
        # model
        rel_complexities = []
        for model in self.model_generator():
            rel_complexities.append(0)
            # Skip models that have already been tested
            if model[MODEL_ID] in self.selection_history:
                continue
            for par in self.parameter_ids:
                rel_par_complexity = self.relative_complexity(
                    old_model[par],
                    float(model[par])
                )
                # Skip models that can not be described as a strict addition
                # (forward selection) or removal (backward selection) of
                # parameters compared to `old_model`.
                if strict:
                    if self.reverse and rel_par_complexity > 0:
                        rel_complexities[-1] = float('nan')
                        break
                    elif not self.reverse and rel_par_complexity < 0:
                        rel_complexities[-1] = float('nan')
                        break
                rel_complexities[-1] += rel_par_complexity
        # If `strict=False` is removed as an option (i.e. `strict` is always
        # `True`), then the comparisons `if i < 0` and `if i > 0` could be
        # removed from the following code.
        if self.reverse:
            next_complexity = max(i for i in rel_complexities if i < 0)
        else:
            next_complexity = min(i for i in rel_complexities if i > 0)
        return [i for i, complexity in enumerate(rel_complexities) if
                complexity == next_complexity]
