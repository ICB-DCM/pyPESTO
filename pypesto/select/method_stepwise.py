import math
from typing import Dict, Iterable, List, Union

import petab

from .problem import ModelSelectionProblem
from .method import ModelSelectorMethod

from .constants import (
    COMPARED_MODEL_ID,
    INITIAL_VIRTUAL_MODEL,
    MODEL_ID,
)

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
            petab_problem: petab.problem,
            model_generator: Iterable[Dict[str, Union[str, float]]],
            criterion: str,
            parameter_ids: List[str],
            selection_history: Dict[str, Dict],
            initial_model: Dict[str, Union[str, float]],
            reverse: bool,
            select_first_improvement: bool,
            startpoint_latest_mle: bool,
            minimize_options: Dict = None,
            criterion_threshold: float = 0,
    ):
        # TODO rename to `default_petab_problem`? There may be multiple petab
        # problems for a single model selection run, defined by the future
        # YAML column
        self.petab_problem = petab_problem
        self.model_generator = model_generator
        self.criterion = criterion
        self.parameter_ids = parameter_ids
        self.selection_history = selection_history
        self.initial_model = initial_model
        self.reverse = reverse
        self.select_first_improvement = select_first_improvement
        self.startpoint_latest_mle = startpoint_latest_mle
        self.minimize_options = minimize_options
        self.criterion_threshold = criterion_threshold

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
        ensure that the model is not considered for selection. TODO valid
        attribute not used at the moment, as `self.initial_model` is now
        implemented.

        TODO:
            fail gracefully if no models are selected after the selection
            algorithm is run with this initial model, so this model is never
            reported as a possible model.
        """

        if self.reverse:
            # TODO ESTIMATE_SYMBOL_INTERNAL
            parameters = dict(zip(self.parameter_ids,
                                  [float("NaN")]*len(self.parameter_ids)))
            # return ModelSelectionProblem(
            #     self.petab_problem,
            #     dict(zip(self.parameter_ids,
            #              [float("NaN")]*len(self.parameter_ids),)),
            #     valid=False
            # )
        else:
            parameters = dict(zip(self.parameter_ids,
                                  [0]*len(self.parameter_ids)))
            # return ModelSelectionProblem(
            #     self.petab_problem,
            #     dict(zip(self.parameter_ids, [0]*len(self.parameter_ids),)),
            #     valid=False
            # )

        model_id = {MODEL_ID: INITIAL_VIRTUAL_MODEL}

        return self.new_model_problem({**model_id, **parameters}, valid=False)
        # return ModelSelectionProblem(
        #     self.petab_problem,
        #     {**model_ID, **parameters},
        #     valid=False
        # )

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
        COMPARED_MODEL_ID.
        """
        selected_models = []
        local_selection_history = {}
        logger.info('%sNew Selection%s', '-'*22, '-'*21)
        # self.setup_direction(self.direction)
        # TODO rewrite so this is in `__init__()`, and this method saves the
        # latest "best" model as `self.model`. Would allow for continuation of
        # from `self.model` by jumping two complexities level above it
        # (assuming forward selection, and that the previous `__call__()`
        # terminated because models that were one complexity above did not
        # produce a better criterion value.
        if self.initial_model is None:
            model = self.new_direction_problem()
        else:
            model = self.new_model_problem(self.initial_model)
            logger.info('Starting with model: %s\n', model.model_id)
            logger.info('Old ID\tNew ID\tCrit\tOld\tNew\tDiff\tResult')
            # copied from for loop -- move into separate function?
            local_selection_history[model.model_id] = {
                MODEL_ID: model.model_id,
                'AIC': model.AIC,
                'BIC': model.BIC,
                # Should this be PYPESTO_INITIAL_MODEL or None? Setting it to
                # PYPESTO_INITIAL_MODEL helps when plotting the directed graph
                # of the selection history
                COMPARED_MODEL_ID: INITIAL_VIRTUAL_MODEL,
                'row': model.row,
                # list, or dict, or zip (iterable)? dict does not preserve
                # order, so is undesirable for use as x_guesses in
                # `pypesto.problem()` (if self.startpoint_latest_mle)
                'MLE': list(zip(
                    model.petab_problem.parameter_df.index,
                    model.optimized_model['x']
                ))
            }
            self.selection_history.update(local_selection_history)
            selected_models.append(local_selection_history[model.model_id])
        proceed = True

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
            compared_model_id = model.model_id
            # compared_model_dict = model.row
            test_models = self.get_test_models(model)
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
            if not test_models and self.initial_model is None:
                raise EOFError('No valid models found.')
            # TODO consider `self.minimize_models(List[ModelSelectionProblem])`
            # and `self.set_minimize_method(List[ModelSelectionProblem])`
            # methods, to allow customisation of the minimize method. The
            # `ModelSelectionProblem` class already has the `autorun` flag
            # to help facilitate this.
            # TODO rewrite loop to select first model that betters the previous
            # model, then move on?
            for test_model_dict in test_models:
                # TODO if the start point of the test model is customised to
                # include MLE from the previous best model, check whether 0
                # will be an issue, since 0 is usually not in the bounds of
                # estimation?
                test_model = self.new_model_problem(
                    test_model_dict,
                    compared_model_id=compared_model_id,
                    # compared_model_dict=compared_model_dict
                )

                local_selection_history[test_model.model_id] = {
                    MODEL_ID: test_model.model_id,
                    'AIC': test_model.AIC,
                    'BIC': test_model.BIC,
                    COMPARED_MODEL_ID: compared_model_id,
                    'row': test_model_dict,
                    'MLE': list(zip(
                        test_model.petab_problem.parameter_df.index,
                        test_model.optimized_model['x']
                    ))
                }

                # TODO necessary to do here? used to exclude models in
                # `ModelSelector.model_generator()`; however, (for example)
                # forward selection would already exclude all models in the
                # `local_selection_history` for being lesser or equal in
                # "complexity" compared to the current best model.
                # Move to after the loop/above the return statement.
                self.selection_history.update(local_selection_history)

                # self.selection_history[test_model.model_id] = {
                #     'AIC': test_model.AIC,
                #     'BIC': test_model.BIC,
                #     'compared_model_id': compared_model_id
                # }

                # The initial model from self.new_direction_problem() is only
                # for complexity comparison, and is not a real model.
                if self.initial_model is None:
                    # TODO bug: if `model` is changed here to `test_model`
                    # then the remaining models will be compared to
                    # `test_model` and not `model`. This will still result in
                    # the correct model being selected, unless
                    # self.criterion_threshold is non-zero, in which case other
                    # models will be required to overcome the threshold
                    # relative to test_model, which may mean models that are
                    # better than this test_model are rejected.
                    model = test_model
                    logger.info('Starting with model: %s\n', model.model_id)
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
                if self.compare(model, test_model):
                    # TODO bug, see bug described in above if statement
                    model = test_model
                    proceed = True
                    if self.select_first_improvement:
                        break

            # could move this to start of loop and check against `model.valid`
            # TODO might be better
            if proceed:
                selected_models.append(local_selection_history[model.model_id])

        # TODO consider changing `selected_models` return value to be a list
        # of the corresponding `ModelSelectionProblem` objects. Might be too
        # memory-intensive, as these objects also contain PEtab and pypesto
        # problem objects.
        return selected_models, local_selection_history, self.selection_history

    def relative_complexity_parameters(self, old: float, new: float) -> int:
        """
        Calculates the relative difference in complexity between two
        parameters. Currently this is as simple as:
        -  0 if both parameters are estimated
        -  0 if both parameters are not estimated
        -  1 if `new` is estimated and `old` is not
        - -1 if `old` is estimated and `new` is not

        TODO rewrite to use self.estimated_parameters to determine whether
             parameters with fixed values should be considered estimated
        TODO "method could be a function"
        """
        return math.isnan(new) - math.isnan(old)
        # # if both parameters are equal "complexity", e.g. both are fixed,
        # # both are estimated.
        # if (math.isnan(old) and math.isnan(new)) or (
        #        not math.isnan(old) and
        #        not math.isnan(new)
        #    ):
        #     return 0
        # # return 1 if the new parameter is estimated, and the old
        # # parameter is fixed
        # elif not math.isnan(old) and math.isnan(new):
        #     return 1
        # # return -1 if the new parameter is fixed, and the old parameter is
        # # estimated
        # elif math.isnan(old) and not math.isnan(new):
        #     return -1

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
            If `True`, then `float('nan')` is returned if the change in the
            "complexity" of any parameter is in the wrong direction, or there
            is no difference in the complexity of the models.
            If `False`, then `float('nan')` is returned if the relative change
            in complexity of the model is in the wrong direction.
            TODO: could be used to instead implement bidirectional selection?
        """
        rel_complexity = 0
        for par in self.parameter_ids:
            rel_par_complexity = math.isnan(model[par])-math.isnan(model0[par])
            # TODO code review: replacement of below code with above code
            # rel_par_complexity = self.relative_complexity_parameters(
            #     model0[par],
            #     model[par]
            # )
            rel_complexity += rel_par_complexity
            # Skip models that can not be described as a strict addition
            # (forward selection) or removal (backward selection) of
            # parameters compared to `model0`.
            # Complexity is set to float('nan') as this value appears to
            # always evaluate to false for comparisons such as a < b.
            # TODO check float('nan') python documentation to confirm
            # TODO code review: which if statement is preferred?
            if self.reverse is None:
                breakpoint()
            if strict and (self.reverse - 0.5)*rel_par_complexity > 0:
                return float('nan')
            # if strict:
            #     if self.reverse and rel_par_complexity > 0:
            #         return float('nan')
            #     elif not self.reverse and rel_par_complexity < 0:
            #         return float('nan')
        if strict and not rel_complexity:
            return float('nan')
        return rel_complexity

    def get_test_models(self,
                        model0_problem: ModelSelectionProblem,
                        strict=True) -> List[int]:
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
            TODO Is `strict=False` useful?
            TODO test strict=False implementation (not implemented)

        Returns
        -------
        A list of indices from `self.model_generator()`, of models that are
        minimally more (for forward selection) or less (for backward selection)
        complex compared to the model described by `model0`.
        """

        # Alternatives
        # a) Currently: a list is generated that contains an element
        # for each model in model_generator, where the element value is the
        # relative complexity of that model compared to model0. Then, the
        # set of indices, of elements with the minimal complexity, in this list
        # is returned.
        # b) loop through models in model generator and keep track of the
        # current minimal complexity change, as well as a list of indices in
        # enumerate(self.model_generator()) that match this minimal complexity.
        # If the next model has a smaller minimal complexity, then replace the
        # current minimal complexity, and replace the list of indices with a
        # list just containing the new model. After the loop, return the list.
        # method alternative (b)
        # TODO rewrite all `model0` and `model` to be `model0_dict` and
        # `model_dict`, then change input argument to `model0`
        minimal_complexity = float('nan')
        test_models = []
        # TODO rewrite `row` attribute to be named `spec` or `specification`
        model0 = model0_problem.row

        # If there exist models that are equal in complexity to the initial
        # model, return them.
        if self.initial_model is None:
            for model in self.model_generator():
                # less efficient than just checking equality in parameters
                # between `model0` and `model`, with `self.parameter_ids`
                if self.relative_complexity_models(
                        model0,
                        model,
                        strict=False) == 0:
                    test_models += [model]
            if test_models:
                return test_models

        # TODO allow for exclusion of already tested models at this point?
        # e.g. if model[MODEL_ID] not in self.selection history. or make new
        # attribute in ModelSelector.model_generator(exclusions: Sequence[str])
        # and call with
        # self.model_generator(exclusions=self.selection_history.keys())
        # then implement exclusion in the generator function...
        for model in self.model_generator():
            model_complexity = self.relative_complexity_models(model0,
                                                               model,
                                                               strict)
            # TODO fix comment after cleanup
            # if model does not represent a valid forward/backward selection
            # option. `isnan` for models with a complexity change in the wrong
            # direction, `not` for models with equivalent complexity.
            # if math.isnan(model_complexity) or not model_complexity:
            if not math.isnan(model_complexity):
                # continue
                if math.isnan(minimal_complexity):
                    minimal_complexity = model_complexity
                    test_models = [model]
            # `abs()` to account for negative complexities in the case of
            # backward selection.
                elif abs(model_complexity) < abs(minimal_complexity):
                    minimal_complexity = model_complexity
                    test_models = [model]
                elif model_complexity == minimal_complexity:
                    test_models += [model]
                else:
                    # TODO remove `continue` after self.initial_model and
                    # related code is implemented
                    continue
                    raise ValueError('Unknown error while calculating '
                                     'relative model complexities.')
        return test_models

        # # method alternative (a)
        # # the nth element in this list is the relative complexity for the nth
        # # model
        # rel_complexities = []
        # for model in self.model_generator():
        #     rel_complexities.append(0)
        #     # Skip models that have already been tested
        #     if model[MODEL_ID] in self.selection_history:
        #         continue
        #     for par in self.parameter_ids:
        #         rel_par_complexity = self.relative_complexity(
        #             model0[par],
        #             model[par]
        #         )
        #         # Skip models that can not be described as a strict addition
        #         # (forward selection) or removal (backward selection) of
        #         # parameters compared to `model0`.
        #         if strict:
        #             if self.reverse and rel_par_complexity > 0:
        #                 rel_complexities[-1] = float('nan')
        #                 break
        #             elif not self.reverse and rel_par_complexity < 0:
        #                 rel_complexities[-1] = float('nan')
        #                 break
        #         rel_complexities[-1] += rel_par_complexity
        # # If `strict=False` is removed as an option (i.e. `strict` is always
        # # `True`), then the comparisons `if i < 0` and `if i > 0` could be
        # # removed from the following code.
        # if self.reverse:
        #     next_complexity = max(i for i in rel_complexities if i < 0)
        # else:
        #     next_complexity = min(i for i in rel_complexities if i > 0)
        # return [i for i, complexity in enumerate(rel_complexities) if
        #         complexity == next_complexity]
