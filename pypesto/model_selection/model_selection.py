import itertools
import tempfile
from typing import Dict, Iterable, List, Sequence, Set, Tuple, Union
from colorama import Fore
import abc
import numpy as np
import math

import petab
from ..problem import Problem
from ..optimize import minimize
from ..result import Result
from ..petab import PetabImporter
from ..objective import Objective

from .constants import (COMPARED_MODEL_ID,
                        ESTIMATE_SYMBOL_INTERNAL,
                        ESTIMATE_SYMBOL_UI,
                        HEADER_ROW,
                        INITIAL_VIRTUAL_MODEL,
                        MODEL_ID,
                        MODEL_ID_COLUMN,
                        NOT_PARAMETERS,
                        PARAMETER_DEFINITIONS_START,
                        PARAMETER_VALUE_DELIMITER,
                        YAML_FILENAME,
                        YAML_FILENAME_COLUMN,)

from petab.C import NOMINAL_VALUE, ESTIMATE


class ModelSelectionProblem:
    """
    Handles the creation, estimation, and evaluation of a model. Here, a model
    is a PEtab problem that is patched with a dictionary of custom parameter
    values (which may specify that the parameter should be estimated).
    Evaluation refers to criterion values such as AIC.
    """
    def __init__(self,
                 row: Dict[str, Union[str, float]],
                 petab_problem: petab.problem,
                 valid: bool = True,
                 autorun: bool = True,
                 x_guess: List[float] = None,
                 x_fixed_estimated: Set[str] = None,
                 minimize_options: Dict = None
    ):
        """
        Arguments
        ---------
        row:
            A single row from the model specification file, in the format that
            is returned by `ModelSelector.model_generator()`.

        petab_problem:
            A petab problem that includes the parameters defined in the model
            specification file.

        valid:
            If `False`, the model will not be tested.

        autorun:
            If `False`, the model parameters will not be estimated. Allows
            users to manually call pypesto.minimize with custom options, then
            `set_result()`.

        x_fixed_estimated:
            TODO not implemented
            Parameters that can be fixed to different values can be considered
            estimated, as the "best" fixed parameter will be preferred. Note,
            the preference is implemented as comparison of different models
            with `ModelSelectionMethod.compare()`, unlike normal estimation,
            which occurs within the same model with `pypesto.minimize`.
        TODO: constraints
        """
        self.row = row
        self.petab_problem = petab_problem
        self.valid = valid

        # TODO may not actually be necessary
        if x_fixed_estimated is None:
            x_fixed_estimated = set()
        else:
            # TODO remove parameters that are zero
            pass

        if minimize_options is None:
            self.minimize_options = {}
        else:
            self.minimize_options = minimize_options

        self.model_id = self.row[MODEL_ID]

        self._AIC = None
        self._BIC = None

        if self.valid:
            # TODO warning/error if x_fixed_estimated is not a parameter ID in
            # the PEtab parameter table. A warning is currently produced in
            # `row2problem` above.
            # Could move to a separate method that is only called when a
            # criterion that requires the number of estimated parameters is
            # called (same for self.n_measurements).
            self.estimated = x_fixed_estimated | set(
                self.petab_problem.parameter_df.query(f'{ESTIMATE} == 1').index
            )
            self.n_estimated = len(self.estimated)
            self.n_measurements = len(petab_problem.measurement_df)

            self.pypesto_problem = row2problem(row,
                                               petab_problem,
                                               x_guess=x_guess)

            self.minimize_result = None

            # TODO autorun may be unnecessary now that the `minimize_options`
            # argument is implemented.
            if autorun:
                if minimize_options:
                    self.set_result(minimize(self.pypesto_problem,
                                             **minimize_options))
                else:
                    self.set_result(minimize(self.pypesto_problem))


    def set_result(self, result: Result):
        self.minimize_result = result
        # TODO extract best parameter estimates, to use as start point for
        # subsequent models in model selection, for parameters in those models
        # that were estimated in this model.
        self.optimized_model = self.minimize_result.optimize_result.list[0]

    @property
    def AIC(self):
        if self._AIC is None:
            self._AIC = calculate_AIC(self.n_estimated,
                                      self.optimized_model.fval)
        return self._AIC

    @property
    def BIC(self):
        if self._BIC is None:
            self._BIC = calculate_BIC(self.n_estimated,
                                      self.n_measurements,
                                      self.optimized_model.fval)
        return self._BIC


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
            #constraints_filename: str
    ):
        self.petab_problem = petab_problem
        # TODO remove duplicates from specification_file
        self.specification_file = unpack_file(specification_filename)
        self.header = line2row(self.specification_file.readline(),
                               convert_parameters_to_float=False)
        self.parameter_ids = self.header[PARAMETER_DEFINITIONS_START:]


        #self.apply_constraints(
        #    self.parse_constraints_file[constraints_filename])

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
        # TODO
        pass

    def apply_constraints(self, constraints: List[Tuple[str, str]]):
        # TODO possible by importing model (also possible petab symbols) into
        # namespace then bool check with sympy
        for model in self.model_generator():
            for constraint_if, constraint_then in constraints:
                pass
        pass

    def model_generator(self,
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

    def select(self,
               method: str,
               criterion: str,
               initial_model: Dict[str, Union[str, float]] = None,
               select_first_improvement: bool = False,
               startpoint_latest_mle: bool = False,
               minimize_options: Dict = None):
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
        """
        if method == 'forward' or method == 'backward':
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
                                       minimize_options)
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
                                           minimize_options)
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
                reverse = False if reverse else True
        elif method == 'all':
            raise NotImplementedError('Testing of all models is not yet '
                                      'implemented.')
        else:
            raise NotImplementedError(f'Model selection algorithm: {method}.')

        # TODO: Reconsider return value. `result` could be stored in attribute,
        # then no values need to be returned, and users can request values
        # manually.
        #return result, self.selection_history
        return selected_models, local_selection_history, self.selection_history


class ModelSelectorMethod(abc.ABC):
    """
    Contains methods that are common to more than one model selection
    algorithm. This is the parent class of model selection algorithms, and
    should not be instantiated.

    Required attributes of child classes are `self.criterion` and
    `self.petab_problem`. `self.minimize_options` should also be set, but can
    be `None`.

    TODO remove `self.petab_problem` once the YAML column rewrite is completed.
    """
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

    def new_model_problem(
            self,
            row: Dict[str, Union[str, float]],
            petab_problem: petab.problem = None,
            valid: bool = True,
            autorun: bool = True,
            compared_model_id: str = None,
            compared_model_dict: str = None,
    ) -> ModelSelectionProblem:
        """
        Creates a ModelSelectionProblem.

        Arguments
        _________
        row:
            A dictionary describing the model, in the format returned by
            `ModelSelector.model_generator()`.
        petab_problem:
            The PEtab problem of the model.
        valid:
            Whether the model should be considered a valid model. If it is not
            valid, it will not be optimized.
        autorun:
            Whether the model should be optimized upon creation.
        compared_model_id:
            The model that new model was compared to. Used to pass the maximum
            likelihood estimate parameters from model `compared_model_id` to
            the current model.
        """
        if petab_problem is None:
            petab_problem = self.petab_problem

        if compared_model_id in self.selection_history:
            # TODO reconsider, might be a bad idea. also removing parameters
            # for x_guess that are not estimated in the new model (as is done)
            # in `row2problem` might also be a bad idea. Both if these would
            # result in x_guess not actually being the latest MLE.
            #if compared_model_dict is None:
            #    raise KeyError('For `startpoint_latest_mle`, the information '
            #                   'of the model that corresponds to the MLE '
            #                   'must be provided. This is to ensure only '
            #                   'estimated parameter values are used in the '
            #                   'startpoint, and all other values are taken '
            #                   'from the PEtab parameter table or the model '
            #                   'specification file.')
            x_guess = self.selection_history[compared_model_id]['MLE']
        else:
            x_guess = None

        return ModelSelectionProblem(
            row,
            self.petab_problem,
            valid=valid,
            autorun=autorun,
            x_guess=x_guess,
            minimize_options=self.minimize_options
        )

    # possibly erroneous now that `ModelSelector.model_generator()` can exclude
    # models, which would change the index of yielded models.
    #def model_by_index(self, index: int) -> Dict[str, Union[str, float]]:
    #    # alternative:
    #    # 
    #    return next(itertools.islice(self.model_generator(), index, None))
    #    #return next(self.model_generator(index=index))

    #def set_exclusions(self, exclusions: List[str])

    #def excluded_models(self,
    #                    exclude_history: bool = True,
    #)

    #def setup_model_generator(self,
    #                          base_model_generator: Generator[
    #                              Dict[str, Union[str, float]],
    #                              None,
    #                              None
    #                          ],
    #) -> None:
    #    self.base_model_generator = base_model_generator

    #def model_generator(self,
    #                    exclude_history: bool = True,
    #                    exclusions: List[str] = None
    #) -> Generator[Dict[str, Union[str, float]], None, None]:
    #    for model in self.base_model_generator():
    #        model_dict = dict(zip(self.header, line2row(line)))
    #        # Exclusion of history makes sense here, to avoid duplicated code
    #        # in specific selectors. However, the selection history of this
    #        # class is only updated when a `selector.__call__` returns, so
    #        # the exclusion of a model tested twice within the same selector
    #        # call is not excluded here. Could be implemented by decorating
    #        # `model_generator` in `ModelSelectorMethod` to include the
    #        # call selection history as `exclusions` (TODO).
    #        if model_dict[MODEL_ID] in exclusions or (
    #                exclude_history and
    #                model_dict[MODEL_ID] in self.selection_history):
    #            continue


class ForwardSelector(ModelSelectorMethod):
    """
    here it is assumed that that there is only one petab_problem
    TODO rewrite `__call__()` here? unsure of intended purpose of separate
    call method that can be called independently/multiple times after
    initialisation...
    """
    def __init__(self,
                 petab_problem: petab.problem,
                 model_generator: Iterable[Dict[str, Union[str, float]]],
                 criterion: str,
                 parameter_ids: List[str],
                 selection_history: Dict[str, Dict],
                 initial_model: Dict[str, Union[str, float]],
                 reverse: bool,
                 select_first_improvement: bool,
                 startpoint_latest_mle: bool,
                 minimize_options: Dict = None
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
            #return ModelSelectionProblem(
            #    self.petab_problem,
            #    dict(zip(self.parameter_ids,
            #             [float("NaN")]*len(self.parameter_ids),)),
            #    valid=False
            #)
        else:
            parameters = dict(zip(self.parameter_ids,
                                  [0]*len(self.parameter_ids)))
            #return ModelSelectionProblem(
            #    self.petab_problem,
            #    dict(zip(self.parameter_ids, [0]*len(self.parameter_ids),)),
            #    valid=False
            #)

        model_id = {MODEL_ID: INITIAL_VIRTUAL_MODEL}

        return self.new_model_problem({**model_id, **parameters}, valid=False)
        #return ModelSelectionProblem(
        #    self.petab_problem,
        #    {**model_ID, **parameters},
        #    valid=False
        #)

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
            #compared_model_dict = model.row
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
                    #compared_model_dict=compared_model_dict
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

                #self.selection_history[test_model.model_id] = {
                #    'AIC': test_model.AIC,
                #    'BIC': test_model.BIC,
                #    'compared_model_id': compared_model_id
                #}

                # The initial model from self.new_direction_problem() is only
                # for complexity comparison, and is not a real model.
                if self.initial_model is None:
                    model = test_model
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

        TODO: rewrite to use self.estimated_parameters to determine whether
              parameters with fixed values should be considered estimated
        """
        # if both parameters are equal "complexity", e.g. both are fixed,
        # both are estimated.
        if (math.isnan(old) and math.isnan(new)) or (
               not math.isnan(old) and
               not math.isnan(new)
           ):
            return 0
        # return 1 if the new parameter is estimated, and the old
        # parameter is fixed
        elif not math.isnan(old) and math.isnan(new):
            return 1
        # return -1 if the new parameter is fixed, and the old parameter is
        # estimated
        elif math.isnan(old) and not math.isnan(new):
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

    def get_test_models(self,
                        #model0: Dict,
                        model0_problem: ModelSelectionProblem,
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
        test_models = []
        # TODO rewrite `row` attribute to be named `spec` or `specification`
        model0 = model0_problem.row

        # If there exist models that are equal in complexity to the initial
        # model, return them.
        if self.initial_model is None:
            for model in self.model_generator():
                # less efficient than just checking equality in parameters
                # between `model0` and `model`, with `self.parameter_ids`
                if self.relative_complexity_models(model0, model) == 0:
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
            model_complexity = self.relative_complexity_models(model0, model)
            # if model does not represent a valid forward/backward selection
            # option. `isnan` for models with a complexity change in the wrong
            # direction, `not` for models with equivalent complexity.
            if math.isnan(model_complexity) or not model_complexity:
                continue
            elif math.isnan(minimal_complexity):
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
                # TODO remove `continue` after self.initial_model and related
                # code is implemented
                continue
                raise ValueError('Unknown error while calculating relative '
                                 'model complexities.')
        return test_models


        ## method alternative (a)
        ## the nth element in this list is the relative complexity for the nth
        ## model
        #rel_complexities = []
        #for model in self.model_generator():
        #    rel_complexities.append(0)
        #    # Skip models that have already been tested
        #    if model[MODEL_ID] in self.selection_history:
        #        continue
        #    for par in self.parameter_ids:
        #        rel_par_complexity = self.relative_complexity(
        #            model0[par],
        #            model[par]
        #        )
        #        # Skip models that can not be described as a strict addition
        #        # (forward selection) or removal (backward selection) of
        #        # parameters compared to `model0`.
        #        if strict:
        #            if self.reverse and rel_par_complexity > 0:
        #                rel_complexities[-1] = float('nan')
        #                break
        #            elif not self.reverse and rel_par_complexity < 0:
        #                rel_complexities[-1] = float('nan')
        #                break
        #        rel_complexities[-1] += rel_par_complexity
        ## If `strict=False` is removed as an option (i.e. `strict` is always
        ## `True`), then the comparisons `if i < 0` and `if i > 0` could be
        ## removed from the following code.
        #if self.reverse:
        #    next_complexity = max(i for i in rel_complexities if i < 0)
        #else:
        #    next_complexity = min(i for i in rel_complexities if i > 0)
        #return [i for i, complexity in enumerate(rel_complexities) if
        #        complexity == next_complexity]

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
            print(Fore.YELLOW + f'Warning: parameter {par_id} is not defined '
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


def calculate_AIC(n_estimated: int, nllh: float) -> float:
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


def calculate_BIC(n_estimated: int, n_measurements: int, nllh: float):
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

