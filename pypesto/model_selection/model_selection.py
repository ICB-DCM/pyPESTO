import itertools
import tempfile
from typing import Dict, Generator, Iterator, List, Tuple, Union

import petab
import numpy as np
import math
from ..problem import Problem
from ..optimize import minimize
from ..result import Result

from .row2problem import row2problem

# TODO move to constants file
MODEL_ID = "modelId"
SBML_FILENAME = "SBML"
# Zero-indexed column indices
MODEL_NAME_COLUMN = 0
SBML_FILENAME_COLUMN = 1
# It is assumed that all columns after PARAMETER_DEFINITIONS_START contain
# parameter Ids.
PARAMETER_DEFINITIONS_START = 2
HEADER_ROW = 0
PARAMETER_VALUE_DELIMITER = ';'
ESTIMATE_SYMBOL_UI = '-'
# here, 'nan' is a string as it will be written to a (temporary) file. The
# actual internal symbol is float('nan'). Equality to this symbol should be
# checked with a function like `math.isnan()` (not ` == float('nan')`).
ESTIMATE_SYMBOL_INTERNAL = 'nan'
INITIAL_VIRTUAL_MODEL = 'PYPESTO_INITIAL_MODEL'


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
            self.n_estimated = sum([1 if is_estimated(p) else 0 \
                                    for header, p in self.row.items() \
                                    if header not in [MODEL_ID, \
                                                      SBML_FILENAME]])
            self.n_edata = None  # TODO for BIC
            self.model_id = self.row[MODEL_ID]
            self.SBML_filename = self.row[SBML_FILENAME]

            self.pypesto_problem = row2problem(row, petab_problem)

            self.minimize_result = None

            if autorun:
                self.set_result(minimize(self.pypesto_problem))

    def set_result(self, result: Result):
        self.minimize_result = result
        self.optimized_model = self.minimize_result.optimize_result.list[0]
        self.AIC = self.calculate_AIC()
        #self.BIC = self.calculate_BIC()

    def calculate_AIC(self):
        # TODO: test, get objective fval for best parameter estimates from
        # minimize_result
        #breakpoint()
        #self.minimize_result.optimize_result.list[i]['fval']

        # TODO double check that fval is negative log likelihood
        return 2*(self.n_estimated + self.optimized_model.fval)
        #return 2*(self.n_estimated - math.log(self.optimized_model.fval))

    def calculate_BIC(self):
        # TODO: test, implement self.n_data in `__init__`
        # TODO: find out how to get size of experimental data
        #       petab.problem
        raise NotImplementedError
        return self.n_estimated*math.log(self.n_data) + \
            2*self.minimize_result[0].fval
            #-2*math.log(self.minimize_result[0].fval)

    def calibrate(self):
        # TODO settings?
        # 100 starts, SingleCoreEngine, ScipyOptimizer
        result = minimize(self.pypesto_problem)
        self.set_result(result)

def is_estimated(p: str) -> bool:
    return math.isnan(p)

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
    """
    expanded_models_file = tempfile.NamedTemporaryFile(mode='r+',
                                                       delete=False)
    with open(file_name) as fh:
        with open(expanded_models_file.name, 'w') as ms_f:
            # could replace `else` condition with ms_f.readline() here, and
            # remove `if` statement completely
            for line_index, line in enumerate(fh):
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

#def seek_to_line(file_object,
#                 line_number: int = 0,
#                 skip_header = True,
#                 relative = False):
#    '''
#    Changes the file pointer to the beginning of a specific line number, either
#    relative to the beginning of the file, or the current line that the file
#    is pointing to.
#
#    Arguments
#    ---------
#    file_object:
#        An open file object that has read access.
#
#    line_number:
#        The requested line number to seek to.
#
#    skip_header:
#        If `True` and `relative == false`, the first line is ignored.
#
#    relative:
#        If `True`, then the current line is considered line 0, otherwise the
#        first line is considered line 0.
#    '''
#    if not relative:
#        file_object.seek(0)
#        if skip_header:
#            file_object.readline()
#    for _ in range(line_number):
#        file_object.readline()


class ModelSelector:
    def __init__(
            self,
            petab_problem: petab.problem,
            specification_filename: str
    ):
        self.petab_problem = petab_problem
        self.specification_file = unpack_file(specification_filename)
        self.header = line2row(self.specification_file.readline(),
                               convert_parameters_to_float=False)
        self.parameter_ids = self.header[PARAMETER_DEFINITIONS_START:]

        self.selection_history = {}

        # Dictionary of method names as keys, with a dictionary as the values.
        # In the dictionary, keys will be modelId, criterion value
        # TODO
        self.results = {}

    def model_generator(self,
                        index: int = None
#    ) -> Union[Dict[str, str], Iterator[Dict[str, str]]]:
    ) -> Generator[Dict[str, Union[str, float]], None, None]:
        """
        A generator for the models described by the model specification file.

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
        #self.specification_file.seek(0)
        #seek_to_line(self.specification_file, 0 if index is None else index)
        #self.specification_file.readline()

        # 1+index to skip the header row
        #seek_to_line(self.specification_file, 0 if index is None else index)
        #print(f'Index in_func: {index}')

        # Go to the start of model specification rows, after the header.
        self.specification_file.seek(0)
        self.specification_file.readline()

        #if index is not None:
        #    print('dictionary:')
        #    print (dict(zip(self.header,
        #                    line2row(self.specification_file.readline()))))
        #    return dict(zip(self.header,
        #                    line2row(self.specification_file.readline())))

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
            return new.AIC < old.AIC
        elif self.criterion == 'BIC':
            return new.BIC < old.BIC
        else:
            raise NotImplementedError('Model selection criterion: '
                                      f'{self.criterion}.')

    def model_by_index(self, index: int) -> Dict[str, Union[str, float]]:
        # alternative:
        # 
        return next(itertools.islice(self.model_generator(), index, None))
        #return next(self.model_generator(index=index))


class ForwardSelector(ModelSelectorMethod):
    """
    here it is assumed that that there is only one petab_problem
    """
    def __init__(self,
                 petab_problem: petab.problem,
                 #model_generator: Iterator[Dict[str, Union[str, float]]],
                 model_generator: Generator[Dict[str, Union[str, float]],
                                            None,
                                            None],
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
        model = self.new_direction_problem()
        proceed = True

        while proceed:
            proceed = False
            # TODO rewrite to use `self.initial_model` later. also consider
            # how initial models for different `__call__()`s should be
            # distinguished.
            # Set here as `model` changes if a better model is found.
            compared_model_id = (INITIAL_VIRTUAL_MODEL
                                 if not model.valid
                                 else model.model_id)
            test_model_indices = self.get_next_step_candidates(model)
            # Error if no valid test models are found. May occur if
            # all models have already been tested
            # TODO rewrite to use a `self.initial_model:bool` attribute
            # or `len(self.selection_history) == 0`. Although,
            # `self.selection_history` may be non-zero from a previously
            # completed `ForwardSelector.__call__()`.
            if not test_model_indices and not model.valid:
                raise Exception('No valid candidate models found.')
            # TODO consider `self.minimize_models(List[ModelSelectionProblem])`
            # and `self.set_minimize_method(List[ModelSelectionProblem])`
            # methods, to allow customisation of the minimize method. The
            # `ModelSelectionProblem` class already has the `autorun` flag
            # to help facilitate this.
            for index in test_model_indices:
                test_model = ModelSelectionProblem(self.petab_problem,
                                                   self.model_by_index(index))
                    #self.model_generator(ind)
                    #row)

                self.selection_history[test_model.model_id] = {
                    'AIC': test_model.AIC,
                    'BIC': test_model.BIC,
                    'compared_model_id': compared_model_id
                }

                # The initial model from self.new_direction_problem() is only
                # for complexity comparison, and is not a real model.
                if not model.valid:
                    model = test_model
                    proceed = True
                    continue

                # at the moment, if multiple models have the "best" criterion
                # value, then the model with the lowest index in
                # ModelSelector.model_generator() is chosen...
                # TODO: this might make visualisation difficult, need a way to
                # predictably select the next model from a set of models that
                # are equivalent by criteria. Alphabetically?
                if self.compare(model, test_model):
                    model = test_model
                    proceed = True

        return model, self.selection_history

    def relative_complexity_parameters(self, old: float, new: float) -> int:
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

    def relative_complexity_models(self,
                                   model0: Dict[str, Union[str, dict]],
                                   model: Dict[str, Union[str, dict]],
                                   strict: bool = True) -> float:
        """
        Calculates the relative different in complexity between two models.
        Models should be in the format returns by
        `ModelSelector.model_generator()`.

        Arguments
        ---------
        model0:
            The model to be compared against.

        model:
            The model to compare.

        strict:
            If `True`, then `float('nan')` is returned if `model` is not a
            a strict increase (for forward selection), or decrease (for
            backward selection), in complexity compared to `model0`. If
            `False`, then only requires a net increase (or decrease) in
            complexity across all parameters. Exception: returns 0 if models
            are equal in complexity.
            TODO: could be used to instead implement bidirectional selection?
        """
        rel_complexity = 0
        for par in self.parameter_ids:
            rel_par_complexity = self.relative_complexity_parameters(
                model0[par],
                model[par]
            )
            rel_complexity += rel_par_complexity
            # Skip models that can not be described as a strict addition
            # (forward selection) or removal (backward selection) of
            # parameters compared to `model0`.
            # Complexity is set to float('nan') as this value appears to
            # always evaluate to false for comparisons such as a < b.
            # TODO check float('nan') python documentation to confirm
            if strict:
                if self.reverse and rel_par_complexity > 0:
                    return float('nan')
                elif not self.reverse and rel_par_complexity < 0:
                    return float('nan')
        return rel_complexity

    def get_next_step_candidates(self,
                                 #model0: Dict,
                                 model0_problem: 'ModelSelectionProblem',
                                 strict=True) -> List[int]:
                                 #conf_dict: Dict[str, float],
                                 #direction=0
        """
        Identifies models are have minimal changes in complexity compared to
        `model0`. Note: models that should be ignored are assigned a
        complexity of `float('nan')`.

        Parameters
        ----------
        model0_problem:
            The model that will be used to calculate the relative complexity of
            other models. TODO: remove the following text after testing new
            implementation: Note: not a `ModelSelectionProblem`, but a
            dictionary in the format that is returned by
            `ModelSelector.model_generator()`.
        strict:
            If `True`, only models that strictly add (for forward selection) or
            remove (for backward selection) parameters compared to `model0`
            will be returned.
            TODO: Is `strict=False` useful?

        Returns
        -------
        A list of indices from `self.model_generator()`, of models that are
        minimally more (for forward selection) or less (for backward selection)
        complex compared to the model described by `model0`.
        """

        '''
        Alternatives
        a) Currently: a list is generated that contains an element
        for each model in model_generator, where the element value is the
        relative complexity of that model compared to model0. Then, the
        set of indices, of elements with the minimal complexity, in this list
        is returned.
        b) loop through models in model generator and keep track of the current
        minimal complexity change, as well as a list of indices in
        enumerate(self.model_generator()) that match this minimal complexity.
        If the next model has a smaller minimal complexity, then replace the
        current minimal complexity, and replace the list of indices with a list
        just containing the new model. After the loop, return the list.
        '''
        # method alternative (b)
        # TODO rewrite all `model0` and `model` to be `model0_dict` and
        # `model_dict`, then change input argument to `model0`
        minimal_complexity = float('nan')
        model_indices = []
        # TODO rewrite `row` attribute to be named `spec` or `specification`
        model0 = model0_problem.row

        # TODO add code here to return any models in model_generator that
        # match the PYPESTO_INITIAL_MODEL, after self.initial_model is
        # implemented, and skip "next model" testing

        # TODO allow for exclusion of already tested models at this point?
        # e.g. if model[MODEL_ID] not in self.selection history. or make new
        # attribute in ModelSelector.model_generator(exclusions: Sequence[str])
        # and call with
        # self.model_generator(exclusions=self.selection_history.keys())
        # then implement exclusion in the generator function...
        for model_index, model in enumerate(self.model_generator()):
            model_complexity = self.relative_complexity_models(model0, model)
            # if model does not represent a valid forward/backward selection
            # option. `isnan` for models with a complexity change in the wrong
            # direction, `not` for models with equivalent complexity.
            if math.isnan(model_complexity) or not model_complexity:
                continue
            elif math.isnan(minimal_complexity):
                minimal_complexity = model_complexity
                model_indices = [model_index]
            # `abs()` to account for negative complexities in the case of
            # backward selection.
            elif abs(model_complexity) < abs(minimal_complexity):
                minimal_complexity = model_complexity
                model_indices = [model_index]
            elif model_complexity == minimal_complexity:
                model_indices += [model_index]
            else:
                # TODO remove `continue` after self.initial_model and related
                # code is implemented
                continue
                raise ValueError('Unknown error while calculating relative '
                                 'model complexities.')
        return model_indices


        # method alternative (a)
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
                    model0[par],
                    model[par]
                )
                # Skip models that can not be described as a strict addition
                # (forward selection) or removal (backward selection) of
                # parameters compared to `model0`.
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
