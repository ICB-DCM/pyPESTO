from typing import Dict, Iterable, List, Set, Sequence, Tuple, Union

import petab

from .method_stepwise import ForwardSelector

from .constants import (
    MODEL_ID,
    PARAMETER_DEFINITIONS_START,
)

from .misc import (
    line2row,
    unpack_file,
)


class ModelSelector:
    """
    Handles model selection. Usage involves initialisation with a model
    specifications file, and then calling the `select()` method to perform
    model selection with a specified algorithm and criterion.
    """
    def __init__(
            self,
            petab_problem: petab.problem,
            specification_filename: str,
            # constraints_filename: str
    ):
        self.petab_problem = petab_problem
        # TODO remove duplicates from specification_file
        self.specification_file = unpack_file(specification_filename)
        self.header = line2row(self.specification_file.readline(),
                               convert_parameters_to_float=False)
        self.parameter_ids = self.header[PARAMETER_DEFINITIONS_START:]

        # self.apply_constraints(
        #     self.parse_constraints_file[constraints_filename])

        self.selection_history = {}

        # Dictionary of method names as keys, with a dictionary as the values.
        # In the dictionary, keys will be modelId, criterion value
        # TODO unused
        self.results = {}

    def delete_if_constraints_fail(self,
                                   model_ids: Set[str] = None,
                                   constraints: Set[Tuple[str, str]] = None):
        """
        TODO method to remove model rows from `self.specification_file` if the
        model matches some constraints

        Arguments
        ---------
        models_ids:
            Rows with these model IDs will be removed.

        constraints:
            Rows that fail any constaints will be removed. The constraints
            be specified as a set of constraints, where each constraint is a
            2-tuple, where the first element is condition that determines
            whether a model should be tested, and the second element is the
            condition that tested models must pass.
        """
        pass

    def parse_constraints_file(
            constraints_filename: str
    ) -> Iterable[Tuple[str, str]]:
        """
        TODO implement and move to misc.py?
        """
        pass

    def apply_constraints(self, constraints: List[Tuple[str, str]]):
        # TODO possible by importing model (also possible petab symbols) into
        # namespace then bool check with sympy
        for model in self.model_generator():
            for constraint_if, constraint_then in constraints:
                pass
        pass

    def model_generator(
            self,
            exclude_history: bool = True,
            exclusions: List[str] = None
    ) -> Iterable[Dict[str, Union[str, float]]]:
        """
        A generator for the models described by the model specification file.

        Argument
        --------
        exclude_history:
            If `True`, models with Id's in `self.selection_history` are not
            yielded.

        exclusions:
            A list of model Id's to avoid yielding.

        Yields
        ------
        Models, one model at a time, as a dictionary, where the keys are the
        column headers in the model specification file, and the values are the
        respective column values in a row of the model specification file.
        """
        # Go to the start of model specification rows, after the header.
        self.specification_file.seek(0)
        self.specification_file.readline()

        if exclusions is None:
            exclusions = []
        if exclude_history:
            exclusions += self.selection_history.keys()

        for line in self.specification_file:
            model_dict = dict(zip(self.header, line2row(line)))
            # Exclusion of history makes sense here, to avoid duplicated code
            # in specific selectors. However, the selection history of this
            # class is only updated when a `selector.__call__` returns, so
            # the exclusion of a model tested twice within the same selector
            # call is not excluded here. Could be implemented by decorating
            # `model_generator` in `ModelSelectorMethod` to include the
            # call selection history as `exclusions` (TODO).
            if model_dict[MODEL_ID] in exclusions:
                continue
            yield model_dict

    # TODO method that automatically generates initial models, for a specific
    # number of starts. TODO parallelise
    def multistart_select(
            self,
            method: str,
            criterion: str,
            initial_models: Sequence[Dict[str, Union[str, float]]] = None,
            select_first_improvement: bool = False,
            startpoint_latest_mle: bool = False,
            minimize_options: Dict = None,
            criterion_threshold: float = 0
    ):
        """
        TODO docstring
        """
        selected_models = []
        local_selection_history = []
        for initial_model in initial_models:
            selected_models_, local_selection_history_, _ = self.select(
                method,
                criterion,
                initial_model=initial_model,
                select_first_improvement=select_first_improvement,
                startpoint_latest_mle=startpoint_latest_mle,
                minimize_options=minimize_options,
                criterion_threshold=criterion_threshold)
            selected_models.append(selected_models_)
            local_selection_history.append(local_selection_history_)

        return selected_models, local_selection_history, self.selection_history

    def select(self,
               method: str,
               criterion: str,
               initial_model: Dict[str, Union[str, float]] = None,
               select_first_improvement: bool = False,
               startpoint_latest_mle: bool = False,
               minimize_options: Dict = None,
               criterion_threshold: float = 0):
        """
        Runs a model selection algorithm. The result is the selected model for
        the current run, independent of previous `select()` calls.

        Arguments
        ---------
        method:
            The model selection algorithm.

        criterion:
            The criterion by which models will be compared.

        initial_model:
            Specify the initial model for the model selection algorithm. If
            `None`, then the algorithm generates an initial model. TODO move
            initial model generation into `ModelSelectionMethod`? Currently,
            `new_model_problem` is there, could be sufficient.
            TODO: reconsider whether input type (dict/ModelSelectionProblem)

        select_first_improvement:
            In forward or backward selection, all models of equally minimally
            greater or less, respectively, complexity compared to the current
            model are optimized, with the best model then selected. If this
            argument is `True`, then the first model that is optimized and
            returns a better criterion value compared to the current model is
            selected, and the other models are not optimized.

        startpoint_latest_mle:
            Specify whether one of the startpoints of the multistart
            optimisation should include the optimized parameters in the current
            model.

        minimize_options:
            A dictionary that will be passed to `pypesto.minimize` as keyword
            arguments for model optimization.

        criterion_threshold:
            The minimum improvement in criterion that a test model must score
            to be selected. The comparison is made according to the method. For
            example, in `ForwardSelector`, test models are compared to the
            previously selected model.
        """
        if method in ('forward', 'backward'):
            reverse = True if method == 'backward' else False
            selector = ForwardSelector(self.petab_problem,
                                       self.model_generator,
                                       criterion,
                                       self.parameter_ids,
                                       self.selection_history,
                                       initial_model,
                                       reverse,
                                       select_first_improvement,
                                       startpoint_latest_mle,
                                       minimize_options,
                                       criterion_threshold)
            result = selector()
            selected_models = result[0]
            local_selection_history = result[1]
            self.selection_history = result[2]
        elif method == 'zigzag':
            # TODO untested
            reverse = False
            selected_models = []
            local_selection_history = {}
            while True:
                selector = ForwardSelector(self.petab_problem,
                                           self.model_generator,
                                           criterion,
                                           self.parameter_ids,
                                           self.selection_history,
                                           initial_model,
                                           reverse,
                                           select_first_improvement,
                                           startpoint_latest_mle,
                                           minimize_options,
                                           criterion_threshold)
                try:
                    result = selector()
                except EOFError:
                    break

                selected_models += result[0]
                local_selection_history.update(result[1])
                self.selection_history.update(result[2])
                # TODO ensure correct functionality with breakpoint here
                breakpoint()
                # TODO consider not setting initial_model to last best model.
                # Then, forward selections from all(theta=0), followed by
                # backward selection from all(theta=estimated), would be
                # repeated until the entire model space is exhausted?
                # TODO include best parameter estimates in initial_model
                # data, for use as startpoint in future tested models
                initial_model = result[0][-1]['row']
                # reverse = False if reverse else True
                reverse = not reverse
        elif method == 'all':
            raise NotImplementedError('Testing of all models is not yet '
                                      'implemented.')
        else:
            raise NotImplementedError(f'Model selection algorithm: {method}.')

        # TODO: Reconsider return value. `result` could be stored in attribute,
        # then no values need to be returned, and users can request values
        # manually.
        # return result, self.selection_history
        return selected_models, local_selection_history, self.selection_history
