from typing import Callable, Dict

import numpy as np

import petab_select
from petab_select import (
    BruteForceCandidateSpace,
    Model,
)
from petab_select.constants import (
    MODEL_ID,
)

from .method import ModelSelectorMethod

import logging
logger = logging.getLogger(__name__)


class BruteForceSelector(ModelSelectorMethod):
    """
    here it is assumed that that there is only one petab_problem
    TODO rewrite `__call__()` here? unsure of intended purpose of separate
    call method that can be called independently/multiple times after
    initialisation...

    Attributes
    ----------
    TODO
    model0:
        A model that has been previously calibrated and available in the
        history `self.selection_history`. The MLE of this calibrated model
        will be used as one start point of the calibration of the other
        models tested in this call.
    """
    def __init__(
            self,
            problem: petab_select.Problem,
            method: str,
            criterion: str,
            selection_history: Dict[str, Dict],
            select_first_improvement: bool,
            startpoint_latest_mle: bool,
            minimize_options: Dict = None,
            # TODO doc, signature in `problem.py:__init__`
            objective_customizer: Callable = None,
            criterion_threshold: float = 0,
            limit: int = np.inf,
            model0: Model = None,
    ):
        self.problem = problem
        self.model_space = problem.model_space
        # FIXME not used for anything
        self.method = method  # should simply be `petab_select.constants.Method.brute_force`?!
        self.criterion = criterion
        self.selection_history = selection_history
        self.select_first_improvement = select_first_improvement
        self.startpoint_latest_mle = startpoint_latest_mle
        self.minimize_options = minimize_options
        self.criterion_threshold = criterion_threshold

        self.limit = limit
        self.candidate_space = BruteForceCandidateSpace()
        self.model0 = model0
        self.objective_customizer = objective_customizer

    def __call__(self):
        """
        Runs the brute force selection algorithm.


        Returns
        -------
        A tuple, where the first element is the selected model, as a
        `ModelSelectionProblem`, and the second element is a dictionary that
        describes the tested models, where the keys are model Ids, and the
        values are dictionaries with the keys 'AIC', 'AICc', 'BIC', and
        COMPARED_MODEL_ID.
        """
        # FIXME specify criteria as call argument such that not all criteria
        #       are calculated every time
        selected_models = []
        local_selection_history = {}
        logger.info('%sNew Selection%s', '-'*22, '-'*21)

        # TODO parallelisation (not sensible if self.select_first_improvement)
        # FIXME implement limit in all selectors
        #test_models = self.model_space.neighbors(
        #    self.candidate_space,
        #    limit=self.limit,
        #)
        calibrated_models = [v['model'] for v in self.selection_history.values()]
        test_models = petab_select.ui.candidates(
            problem=self.problem,
            candidate_space=self.candidate_space,
            limit=self.limit,
            excluded_models=calibrated_models,
        ).models
        # Error if no valid test models are found. May occur if
        # all models have already been tested. `Exception` may be a bad way
        # to handle this... a warning?
        # TODO need to change this check to be whether any models were
        # successfully selected.
        # if not test_models and self.initial_model is None:  FIXME
        if not test_models and not selected_models:
            raise StopIteration('No valid models found.')
        # TODO consider `self.minimize_models(List[ModelSelectionProblem])`
        # and `self.set_minimize_method(List[ModelSelectionProblem])`
        # methods, to allow customisation of the minimize method. The
        # `ModelSelectionProblem` class already has the `autorun` flag
        # to help facilitate this.
        for test_model in test_models:
            # TODO if the start point of the test model is customised to
            # include MLE from the previous best model, check whether 0
            # will be an issue, since 0 is usually not in the bounds of
            # estimation?
            test_model_problem = self.new_model_problem(
                model=test_model,
                criterion=self.criterion,
                #model0=self.model0,
                # model0=model_problem.model,  FIXME
            )

            #test_model_problem.compute_all_criteria()
            #test_model.set_criterion(
            #    self.criterion,
            #    test_model_problem.get_criterion(self.criterion),
            #)

            # FIXME Change dictionary to `petab_select.model.Model` object
            # instead.
            local_selection_history[test_model.model_id] = {
                MODEL_ID: test_model.model_id,
                'model': test_model,
                'model0': self.model0,
                # COMPARED_MODEL_ID: compared_model_id,  FIXME
                'MLE': dict(zip(
                    #test_model_problem.petab_problem.parameter_df.index,
                    test_model_problem.pypesto_problem.x_names,
                    test_model_problem.optimized_model['x']
                ))
            }

            # TODO necessary to do here? used to exclude models in
            # `ModelSelector.model_generator()`; however, (for example)
            # forward selection would already exclude all models in the
            # `local_selection_history` for being lesser or equal in
            # "complexity" compared to the current best model.
            # Move to after the loop/above the return statement.
            self.selection_history.update(local_selection_history)

            # at the moment, if multiple models have the "best" criterion
            # value, then the model with the lowest index in
            # ModelSelector.model_generator() is chosen...
            # TODO: this might make visualisation difficult, need a way to
            # predictably select the next model from a set of models that
            # are equivalent by criteria. Alphabetically?
            # FIXME switch to `petab_select.Problem.compare`
            if (
                self.select_first_improvement and
                self.model0 is not None and
                self.problem.compare(self.model0, test_model)
            ):
                break

        # could move this to start of loop and check against `model.valid`
        # TODO might be better
        selected_models.append(self.problem.get_best(test_models))

        # TODO consider changing `selected_models` return value to be a list
        # of the corresponding `ModelSelectionProblem` objects. Might be too
        # memory-intensive, as these objects also contain PEtab and pypesto
        # problem objects.
        return selected_models, local_selection_history, self.selection_history
