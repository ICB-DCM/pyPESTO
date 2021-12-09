import abc
from typing import Dict, Iterable, List, Optional, Union

from more_itertools import one
import numpy as np

import petab_select
from petab_select import (
    BACKWARD,
    BIDIRECTIONAL,
    BRUTE_FORCE,
    FORWARD,
    Model,
)

from .constants import TYPE_POSTPROCESSOR
from .method_brute_force import BruteForceSelector
from .method_stepwise import ForwardSelector


class ModelSelector(abc.ABC):
    """Handles use of a model selection algorithm.

    Handles model selection. Usage involves initialisation with a model
    specifications file, and then calling the `select()` method to perform
    model selection with a specified algorithm and criterion.
    """

    def __init__(
        self,
        problem: petab_select.Problem,
        model_postprocessor: Optional[TYPE_POSTPROCESSOR] = None,
    ):
        self.problem = problem
        self.model_space = self.problem.model_space

        self.model_postprocessor = model_postprocessor

        self.selection_history = {}

        # Dictionary of method names as keys, with a dictionary as the values.
        # In the dictionary, keys will be modelId, criterion value
        # TODO unused
        self.results = {}

    # TODO method that automatically generates initial models, for a specific
    # number of starts. TODO parallelise
    def multistart_select(
        self,
        method: str,
        criterion: str,
        initial_models: Iterable[Model] = None,
        select_first_improvement: bool = False,
        startpoint_latest_mle: bool = False,
        minimize_options: Dict = None,
        criterion_threshold: float = 0,
    ):
        """Run an algorithm multiple times, with different initial models."""
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
                criterion_threshold=criterion_threshold,
            )
            selected_models.append(selected_models_)
            local_selection_history.append(local_selection_history_)

        return selected_models, local_selection_history, self.selection_history

    def select_to_completion(
        self,
        *args,
        **kwargs,
    ) -> List[Model]:
        """Run an algorithm until it terminates."""
        history_of_best_model = []
        while True:
            if history_of_best_model:
                kwargs["initial_model"] = history_of_best_model[-1]
            try:
                (selected_models, _, self.selection_history,) = self.select(
                    *args,
                    **kwargs,
                )
                history_of_best_model.append(one(selected_models))
            except StopIteration:
                break

        return history_of_best_model

    def select(
        self,
        method: str = None,
        criterion: str = None,
        initial_model: Model = None,
        select_first_improvement: bool = False,
        startpoint_latest_mle: bool = False,
        minimize_options: Dict = None,
        # TODO doc, signature in `problem.py:__init__`
        objective_customizer: Dict = None,
        criterion_threshold: float = 0,
        limit: Union[float, int] = np.inf,
    ):
        """Run a single iteration of a model selection algorithm.

        The result is the selected model for the current run, independent of
        previous selected models.

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

        limit:
            Limit the number of tested models.
        """
        # TODO move some options to PEtab Select? e.g.:
        # - startpoint_latest_mle
        # - select_first_improvement
        if method is None:
            method = self.problem.method
        if criterion is None:
            criterion = self.problem.criterion
        if method in (FORWARD, BACKWARD):
            reverse = True if method == BACKWARD else False
            selector = ForwardSelector(
                problem=self.problem,
                method=method,
                criterion=criterion,
                selection_history=self.selection_history,
                initial_model=initial_model,
                reverse=reverse,
                select_first_improvement=select_first_improvement,
                startpoint_latest_mle=startpoint_latest_mle,
                minimize_options=minimize_options,
                criterion_threshold=criterion_threshold,
                objective_customizer=objective_customizer,
                limit=limit,
                model_postprocessor=self.model_postprocessor,
            )
            result = selector()
            selected_models = result[0]
            local_selection_history = result[1]
            self.selection_history = result[2]
        elif method == BIDIRECTIONAL:
            # TODO untested
            reverse = False
            selected_models = []
            local_selection_history = {}
            while True:
                raise NotImplementedError("FIXME: edit to match new format")
                selector = ForwardSelector(
                    self.model_space,
                    method,
                    criterion,
                    self.selection_history,
                    initial_model,
                    reverse,
                    select_first_improvement,
                    startpoint_latest_mle,
                    minimize_options,
                    criterion_threshold,
                    objective_customizer=objective_customizer,
                    limit=limit,
                    model_postprocessor=self.model_postprocessor,
                )
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
                initial_model = result[0][-1]["row"]
                # reverse = False if reverse else True
                reverse = not reverse
        elif method == BRUTE_FORCE:
            selector = BruteForceSelector(
                problem=self.problem,
                method=method,
                # remove option?
                criterion=criterion,
                selection_history=self.selection_history,
                select_first_improvement=select_first_improvement,
                startpoint_latest_mle=startpoint_latest_mle,
                minimize_options=minimize_options,
                objective_customizer=objective_customizer,
                criterion_threshold=criterion_threshold,
                limit=limit,
                # TODO rename to model0
                model0=initial_model,
                model_postprocessor=self.model_postprocessor,
            )
            (
                selected_models,
                local_selection_history,
                _,
            ) = selector()
        else:
            raise NotImplementedError(f"Model selection algorithm: {method}.")

        # TODO: Reconsider return value. `result` could be stored in attribute,
        # then no values need to be returned, and users can request values
        # manually.
        # return result, self.selection_history
        return selected_models, local_selection_history, self.selection_history
