from typing import Callable, Dict, Optional, Union

import petab_select
from petab_select import (
    BackwardCandidateSpace,
    ForwardCandidateSpace,
    Model,
    VIRTUAL_INITIAL_MODEL
)
from petab_select.constants import (
    MODEL_ID,
)

from .constants import TYPE_POSTPROCESSOR
from .method import ModelSelectorMethod

import logging
logger = logging.getLogger(__name__)


class ForwardSelector(ModelSelectorMethod):
    """
    here it is assumed that that there is only one petab_problem
    TODO rewrite `__call__()` here? unsure of intended purpose of separate
    call method that can be called independently/multiple times after
    initialisation...
    """
    def __init__(
            self,
            problem: petab_select.Problem,
            method: str,
            criterion: str,
            selection_history: Dict[str, Dict],
            initial_model: Union[Model, None],
            reverse: bool,
            select_first_improvement: bool,
            startpoint_latest_mle: bool,
            minimize_options: Dict = None,
            criterion_threshold: float = 0,
            objective_customizer: Callable = None,
            limit: int = None,
            model_postprocessor: Optional[TYPE_POSTPROCESSOR] = None,
    ):
        # TODO rename to `default_petab_problem`? There may be multiple petab
        # problems for a single model selection run, defined by the future
        # YAML column
        self.problem = problem
        self.model_space = problem.model_space
        self.method = method
        self.criterion = criterion
        self.selection_history = selection_history
        self.initial_model = initial_model
        self.reverse = reverse
        self.select_first_improvement = select_first_improvement
        self.startpoint_latest_mle = startpoint_latest_mle
        self.minimize_options = minimize_options
        self.criterion_threshold = criterion_threshold

        # TODO expect that VIRTUAL_INITIAL_MODEL is passed by the user or at
        #      least set further upstream than here?
        if self.initial_model is None:
            self.initial_model = VIRTUAL_INITIAL_MODEL
        if reverse:
            self.candidate_space = BackwardCandidateSpace()
        else:
            self.candidate_space = ForwardCandidateSpace()

        self.objective_customizer = objective_customizer
        self.limit = limit
        self.model_postprocessor = model_postprocessor

    def __call__(self, predecessor_model: Model = None):
        """
        Runs the forward (or backward, if `self.reverse=True`) selection
        algorithm.

        Arguments
        ---------
        predecessor_model:
            Candidate models will be sought in the neighborhood of this
            model.

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
        # FIXME proceed now means "do full forward selection"
        proceed = True
        break_after_first_iteration = True

        if predecessor_model is None:
            predecessor_model = self.initial_model

        # TODO parallelisation (not sensible if self.select_first_improvement)
        # TODO rename `proceed` to `improved_criterion`
        while proceed:
            proceed = False
            # TODO how should initial models for different `__call__()`s be
            # distinguished (or, different `ModelSelector.select()` calls...)
            # Set here as `model` changes if a better model is found. TODO no
            # longer necessary if "first better test model is chosen" is the
            # only algorithm, not "all test models are compared, best test
            # model is chosen".
            calibrated_models = \
                [v['model'] for v in self.selection_history.values()]
            test_models = petab_select.ui.candidates(
                problem=self.problem,
                candidate_space=self.candidate_space,
                initial_model=predecessor_model,
                # TODO use PEtab Select to get best model from previous
                #      iteration, to use as initial model, instead of
                #      providing?
                limit=self.limit,
                excluded_models=calibrated_models,
            ).models
            # Error if no valid test models are found. May occur if
            # all models have already been tested. `Exception` may be a bad way
            # to handle this... a warning?
            # Currently, not `ModelSelectionProblem.valid` is not used now that
            # `self.initial_model` is implemented; however, `valid` may be used
            # later and replace `self.initial_model` here (assuming
            # `self.initial_model.valid == False`)
            # TODO now that initial models can be specified, rename
            # self.initial_model to self.initial_virtual_model? Also, need to
            # change this check to be whether any models were successfully
            # selected.
            if not test_models and not selected_models:
                raise StopIteration('No valid models found.')
            # TODO consider `self.minimize_models(List[ModelSelectionProblem])`
            # and `self.set_minimize_method(List[ModelSelectionProblem])`
            # methods, to allow customisation of the minimize method. The
            # `ModelSelectionProblem` class already has the `autorun` flag
            # to help facilitate this.
            # TODO rewrite loop to select first model that betters the previous
            # model, then move on?
            for test_model in test_models:
                # TODO if the start point of the test model is customised to
                # include MLE from the previous best model, check whether 0
                # will be an issue, since 0 is usually not in the bounds of
                # estimation?
                test_model_problem = self.new_model_problem(
                    model=test_model,
                    criterion=self.criterion,
                )

                # TODO: refactor to simply be `Model.model_id : Model`
                local_selection_history[test_model.model_id] = {
                    MODEL_ID: test_model.model_id,
                    'model': test_model,
                    'MLE': dict(zip(
                        test_model_problem.pypesto_problem.x_names,
                        test_model_problem.optimized_model['x']
                    )),
                }

                # TODO necessary to do here? used to exclude models in
                # `ModelSelector.model_generator()`; however, (for example)
                # forward selection would already exclude all models in the
                # `local_selection_history` for being lesser or equal in
                # "complexity" compared to the current best model.
                # Move to after the loop/above the return statement.
                self.selection_history.update(local_selection_history)

                if predecessor_model == VIRTUAL_INITIAL_MODEL:
                    # TODO bug: if `model` is changed here to `test_model`
                    # then the remaining models will be compared to
                    # `test_model` and not `model`. This will still result in
                    # the correct model being selected, unless
                    # self.criterion_threshold is non-zero, in which case other
                    # models will be required to overcome the threshold
                    # relative to test_model, which may mean models that are
                    # better than this test_model are rejected.
                    model_problem = test_model_problem
                    logger.info(
                        'Starting with model: %s\n',
                        model_problem.model_id,
                    )
                    logger.info('Old ID\tNew ID\tCrit\tOld\tNew\tDiff\tResult')
                    # TODO reconsider whether `False` is appropriate, after
                    # refactor that changed self.initial_model to be None if
                    # no initial model (as a dict) is specified. Could change
                    # to empty dict()?
                    self.initial_model = False
                    proceed = True
                    continue

                # at the moment, if multiple models have the "best" criterion
                # value, then the model with the lowest index in
                # ModelSelector.model_generator() is chosen...
                # TODO: this might make visualisation difficult, need a way to
                # predictably select the next model from a set of models that
                # are equivalent by criteria. Alphabetically?
                if self.compare(predecessor_model, test_model):
                    # TODO bug, see bug described in above if statement
                    proceed = True
                    if self.select_first_improvement:
                        break

            local_models = \
                [item['model'] for item in local_selection_history.values()]
            best_local_model = \
                self.problem.get_best(local_models, self.criterion)
            selected_models.append(best_local_model)

            if break_after_first_iteration:
                break

        # TODO consider changing `selected_models` return value to be a list
        # of the corresponding `ModelSelectionProblem` objects. Might be too
        # memory-intensive, as these objects also contain PEtab and pypesto
        # problem objects.
        return selected_models, local_selection_history, self.selection_history
