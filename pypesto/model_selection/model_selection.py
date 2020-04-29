import itertools
import tempfile
from typing import Dict, Iterator, List, Tuple

import petab
import numpy as np
from ..problem import Problem

MODEL_ID = "ModelId"
# Zero-indexed column indices
MODEL_NAME_COLUMN = 0
SBML_FILENAME_COLUMN = 1
PARAMETER_DEFINITIONS_START = 2
HEADER_ROW = 0
PARAMETER_VALUE_DELIMITER = ';'
ESTIMATE_SYMBOL_UI = '-'
# here, 'nan' is a string as it will be written to a file. The actual
# internal symbol is float('nan')
ESTIMATE_SYMBOL_INTERNAL = 'nan'

class ModelSelectionProblem:
    """

    """
    def __init__(self,
                 petab_problem: petab.problem,
                 row: Dict[str, float],
                 valid: bool = True
    ):
        '''
        Valid argument is true if the model is in the model specifications
        file, and does not violate any constraints.
        TODO: constraints
        '''
        self.pypesto_problem = None  # self.row2problem(petab_problem, row)
        self.optimization_result = None
        self.model_id = row[MODEL_ID]
        self.AIC = None
        self.BIC = None
        self.configuration = row
        self.valid = valid

    def get_aic(self):
        raise NotImplementedError()

    def get_bic(self):
        raise NotImplementedError()

    def row2problem(self, petab_problem, row) -> Problem:
        # in progress
        raise NotImplementedError()

    def calibrate(self):
        # TODO
        # self.get_aic()
        # self.get_bic()
        self.AIC = self.configuration['AIC']

def _replace_estimate_symbol(parameter_definition) -> List:
    return [ESTIMATE_SYMBOL_INTERNAL if p == ESTIMATE_SYMBOL_UI else p
            for p in parameter_definition]

def unpack_file(file_name: str):
    '''
    Unpacks a model specification file into a new temporary file, which is
    returned.

    TODO
        - Consider alternatives to `_{n}` suffix for model `modelId`
        - How should the selected model be reported to the user? Remove the
          `_{n}` suffix and report the original `modelId` alongside the
          selected parameters? Generate a set of PEtab files with the
          chosen SBML file and the parameters specified in a parameter or
          condition file?
    '''
    expanded_models_file = tempfile.NamedTemporaryFile(mode='w',
                                                       delete=False)
    with open(file_name) as fh:
        for line_index, line in enumerate(fh):
            if line_index != HEADER_ROW:
                columns = ModelSelector.line2row(line)
                parameter_definitions = [
                    _replace_estimate_symbol(
                        definition.split(PARAMETER_VALUE_DELIMITER)
                    ) for definition in columns[
                        PARAMETER_DEFINITIONS_START:
                    ]
                ]
                for index, selection in enumerate(itertools.product(
                        *parameter_definitions
                )):
                    expanded_models_file.write(
                        '\t'.join([
                            columns[MODEL_NAME_COLUMN]+f'_{index}',
                            columns[SBML_FILENAME_COLUMN],
                            *selection
                        ]) + '\n'
                    )
            else:
                expanded_models_file.write(line)
    return expanded_models_file

def line2row(line: str, delimiter: str = '\t') -> List:
    return line.strip().split(delimiter)

class ModelSelector:
    def __init__(
            self,
            petab_problem: petab.problem,
            specification_filename: str
    ) -> ModelSelector:
        self.petab_problem = petab_problem
        self.specification_file = \
            ModelSelectionHelper.unpack_file(specification_filename)
        self.header = line2row(self.specification_file.readline())
        self.parameter_ids = self.header[PARAMETER_DEFINITIONS_START:]

        self.selection_history = []

        # Dictionary of method names as keys, with a dictionary as the values.
        # In the dictionary, keys will be modelId, criterion value
        # TODO
        self.results = {}

    def model_generator(self, index: int = None) -> Iterator[Dict[str, str]]:
        # If index is specified, return the specific model
        # 1+index to skip the header row
        if index is not None:
            self.specification_file.seek(1+index)
            return dict(zip(self.header,
                            line2row(next(self.specification_file))))

        # Go to start of file, after header
        self.specification_file.seek(1)
        for line in self.specification_file:
            yield dict(zip(self.header, line2row(line)))

    def select(self, method: str, criterion: str):
        if method == 'forward':
            selector = ForwardSelector(self.petab_problem,
                                       self.model_generator,
                                       criterion,
                                       self.parameter_ids
                                       self.selection_history)
            result, self.selection_history = selector()
        if method == 'backward':
            selector = ForwardSelector(self.petab_problem,
                                       self.model_generator,
                                       criterion,
                                       self.parameter_ids,
                                       self.selection_history,
                                       reverse=True)
            result, self.selection_history = selector()

class ModelSelectorMethod:
    # the calling child class should have self.criterion defined
    def compare(self, old: ModelSelectionProblem, new: ModelSelectionProblem):
        if self.criterion == 'AIC':
            return new.get_aic() < old.get_aic()

class ForwardSelector(ModelSelectorMethod):
    """
    here it is assumed that that there is only one petab_problem
    """
    def __init__(self,
                 petab_problem: petab.problem,
                 model_generator: Iterator[Dict[str, str]],
                 criterion: str,
                 parameter_ids: List[str],
                 selection_history: List[str],
                 reverse: bool = False)
        self.petab_problem = petab_problem
        self.model_generator = model_generator
        self.criterion = criterion
        self.parameter_ids = parameter_ids
        self.selection_history = selection_history
        self.reverse = reverse

    def new_direction_problem(self) -> ModelSelectionProblem:
        # This specifies an initial model where all parameters are estimated
        # (for backward selection) or all parameters are zero (for forward
        # selection). This "model" will never be simulated/tested, just used
        # as a point of comparison.
        # TODO: fail gracefully if no models are selected after the selection
        # algorithm is run with this initial model, so this model is never
        # reported as a possible model.
        if self.reverse:
            return ModelSelectionProblem(
                self.petab_problem,
                dict(zip(
                    self.parameter_ids,
                    [ESTIMATE_SYMBOL_INTERNAL for _ in len(self.parameter_ids)]
                )),
                valid=False
            )
        else:
            return ModelSelectionProblem(
                self.petab_problem,
                # Could be [0]*len(self.parameter_ids)
                dict(zip(
                    self.parameter_ids,
                    [0 for _ in len(self.parameter_ids)],
                )),
                valid=False
            )

    def __call__(self):
        self.setup_direction(self.direction)
        ms_problem = self.new_direction_problem()
        proceed = True

        while proceed:
            proceed = False
            model_candidate_indices = self.get_next_step_candidates(
                ms_problem.configuration)
            # Error if no valid candidate models are found. May occur if
            # all models have already been tested
            if not model_candidate_indices and not ms_problem.valid:
                raise Exception('No valid candidate models found.')
            better_model_id = None
            for ind in model_candidate_indices:
                candidate_model = ModelSelectionProblem(
                    self.petab_problem,
                    self.model_generator(ind))
                    #self.petab_problem, self.unpacked_ms_conf[ind])
                candidate_model.calibrate()
                self.selection_history.append(candidate_model.model_id)

                # The initial model from self.new_direction_problem() is only
                # for complexity comparison, and is not a real model.
                if not ms_problem.valid:
                    ms_problem = candidate_model
                    better_model_id = candidate_model.model_id
                    proceed = True
                    continue

                if self.compare(ms_problem, candidate_model):
                    ms_problem = candidate_model
                    better_model_id = candidate_model.model_id
                    proceed = True

        return ms_problem, self.selection_history

    def relative_complexity(self, old: float, new: float) -> int:
        '''
        Currently, non-zero fixed values are considered equal in complexity to
        estimated values.
        '''
        # if both parameters are equal "complexity", e.g. both are zero,
        # both are estimated, or both are non-zero values.
        if (old == new == 0) or \
           (math.isnan(old) and math.isnan(new)) or (
               not math.isnan(old) and \
               not math.isnan(new) and \
               old != 0 and \
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
                                 #conf_dict: Dict[str, float],
                                 #direction=0
    ) -> List[int]:
        """
        returns indices of models that should be considered on the next step
        for now just direction = 0 is considered

        Parameters
        ----------
        conf_dict:
        direction: 0 - forward, 1 - backward

        Returns
        -------

        """
        # the nth element in this list is the relative complexity for the nth
        # model
        if 
        rel_complexity_orders = []
        for model_descr in self.model_generator():
            rel_complexity_orders.append(0)
            # Skip models that have already been tested
            if model_descr.model_id in self.selection_history:
                continue
            for par in self.parameter_ids:
                rel_complexity_orders[-1] += relative_complexity(
                    conf_dict[par],
                    model_descr[par]
                )
        if self.reverse:
            next_complexity = max(i for i in rel_complexity_orders if i < 0)
        else:
            next_complexity = min(i for i in rel_complexity_orders if i > 0)
        return [i for i, complexity in enumerate(rel_complexity_orders) if
                complexity == next_complexity]
