from abc import ABC, abstractmethod
from typing import Dict, Union

import petab

from .problem import ModelSelectionProblem

import logging
logger = logging.getLogger(__name__)


class ModelSelectorMethod(ABC):
    """
    Contains methods that are common to more than one model selection
    algorithm. This is the parent class of model selection algorithms, and
    should not be instantiated.

    Required attributes of child classes are `self.criterion` and
    `self.petab_problem`. `self.minimize_options` should be set, but can be
    `None`. `self.criterion_threshold` should be set, but can be zero.

    TODO remove `self.petab_problem` once the YAML column rewrite is completed.
    """
    # Described in the docstring for the `ModelSelector.select` method.
    criterion: str
    # Described in the docstring for the `ModelSelector.select` method.
    criterion_threshold: float
    # TODO docstring
    petab_problem: petab.Problem
    # TODO docstring
    selection_history: Dict[str, Dict]
    # TODO docstring
    minimize_options: Dict

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
        # TODO implement criterion as @property of ModelSelectorMethod
        if self.criterion == 'AIC':
            result = new.AIC + self.criterion_threshold < old.AIC
            logger.info('%s\t%s\tAIC\t%.3f\t%.3f\t%.3f\t%s',
                        old.model_id,
                        new.model_id,
                        old.AIC,
                        new.AIC,
                        new.AIC-old.AIC,
                        "Accepted" if result else "Rejected")
            # logger.info(f'{old.model_id}\t{new.model_id}\tAIC\t{old.AIC:.3f}\t'
            #             f'{new.AIC:.3f}\t'
            #             f'{new.AIC-old.AIC:.3f}\t'
            #             f'{"Accepted" if result else "Rejected"}')
            # return result
        elif self.criterion == 'BIC':
            result = new.BIC + self.criterion_threshold < old.BIC
            logger.info('%s\t%s\tAIC\t%.3f\t%.3f\t%.3f\t%s',
                        old.model_id,
                        new.model_id,
                        old.BIC,
                        new.BIC,
                        new.BIC-old.BIC,
                        "Accepted" if result else "Rejected")
            # logger.info(f'{old.model_id}\t{new.model_id}\tBIC\t{old.BIC:.3f}\t'
            #             f'{new.BIC:.3f}\t'
            #             f'{new.BIC-old.BIC:.3f}\t'
            #             f'{"Accepted" if result else "Rejected"}')
            # return result
        else:
            raise NotImplementedError('Model selection criterion: '
                                      f'{self.criterion}.')
        return result

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
            # if compared_model_dict is None:
            #     raise KeyError('For `startpoint_latest_mle`, the information'
            #                    ' of the model that corresponds to the MLE '
            #                    'must be provided. This is to ensure only '
            #                    'estimated parameter values are used in the '
            #                    'startpoint, and all other values are taken '
            #                    'from the PEtab parameter table or the model '
            #                    'specification file.')
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
    # def model_by_index(self, index: int) -> Dict[str, Union[str, float]]:
    #     # alternative:
    #     #
    #     return next(itertools.islice(self.model_generator(), index, None))
    #     #return next(self.model_generator(index=index))

    # def set_exclusions(self, exclusions: List[str])

    # def excluded_models(self,
    #                     exclude_history: bool = True,
    # )

    # def setup_model_generator(self,
    #                           base_model_generator: Generator[
    #                               Dict[str, Union[str, float]],
    #                               None,
    #                               None
    #                           ],
    # ) -> None:
    #     self.base_model_generator = base_model_generator

    # def model_generator(self,
    #                     exclude_history: bool = True,
    #                     exclusions: List[str] = None
    # ) -> Generator[Dict[str, Union[str, float]], None, None]:
    #     for model in self.base_model_generator():
    #         model_dict = dict(zip(self.header, line2row(line)))
    #         # Exclusion of history makes sense here, to avoid duplicated code
    #         # in specific selectors. However, the selection history of this
    #         # class is only updated when a `selector.__call__` returns, so
    #         # the exclusion of a model tested twice within the same selector
    #         # call is not excluded here. Could be implemented by decorating
    #         # `model_generator` in `ModelSelectorMethod` to include the
    #         # call selection history as `exclusions` (TODO).
    #         if model_dict[MODEL_ID] in exclusions or (
    #                 exclude_history and
    #                 model_dict[MODEL_ID] in self.selection_history):
    #             continue
