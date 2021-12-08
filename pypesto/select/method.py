import abc
import logging
from typing import Dict

import petab
from petab_select.constants import (
    VIRTUAL_INITIAL_MODEL,
)
from petab_select import (
    Model,
    Criterion,
    #AIC,
    #AICC,
    #BIC,
)

from .problem import ModelSelectionProblem


logger = logging.getLogger(__name__)


class ModelSelectorMethod(abc.ABC):
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

    # TODO general __init__ that sets e.g. model_postprocessor

    # the calling child class should have self.criterion defined
    def compare(
        self,
        old_model: Model,
        new_model: Model,
    ) -> bool:
        """
        Compares models by criterion.

        Arguments
        ---------
        old_model:
            A calibrated model
        new_model:
            A calibrated model.

        Returns
        -------
        `True`, if `new_model` is superior to `old_model` by the criterion,
        else `False`.
        """
        # TODO switch to `petab_select.model.default_compare`
        # should then allow for easy extensibility to compare custom criteria
        if self.criterion in [
            Criterion.AIC,
            Criterion.AICC,
            Criterion.BIC,
        ]:
            new_criterion = new_model.get_criterion(self.criterion)
            old_criterion = old_model.get_criterion(self.criterion)
            result = new_criterion + self.criterion_threshold < old_criterion
            logger.info(
                '%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%s',
                old_model.model_id,
                new_model.model_id,
                self.criterion,
                old_criterion,
                new_criterion,
                new_criterion - old_criterion,
                "Accepted" if result else "Rejected",
            )
        ## TODO implement criterion as @property of ModelSelectorMethod
        ## TODO refactor to reduce repeated code
        #if self.criterion == 'AIC':
        #    result = new.aic + self.criterion_threshold < old.aic
        #    logger.info('%s\t%s\tAIC\t%.3f\t%.3f\t%.3f\t%s',
        #                old.model_id,
        #                new.model_id,
        #                old.aic,
        #                new.aic,
        #                new.aic - old.aic,
        #                "Accepted" if result else "Rejected")
        #    # logger.info(f'{old.model_id}\t{new.model_id}\tAIC\t{old.AIC:.3f}\t'
        #    #             f'{new.AIC:.3f}\t'
        #    #             f'{new.AIC-old.AIC:.3f}\t'
        #    #             f'{"Accepted" if result else "Rejected"}')
        #    # return result
        #elif self.criterion == 'AICc':
        #    result = new.aicc + self.criterion_threshold < old.aicc
        #    logger.info('%s\t%s\tAICc\t%.3f\t%.3f\t%.3f\t%s',
        #                old.model_id,
        #                new.model_id,
        #                old.aicc,
        #                new.aicc,
        #                new.aicc - old.aicc,
        #                "Accepted" if result else "Rejected")
        #    # logger.info(f'{old.model_id}\t{new.model_id}\tAIC\t{old.AIC:.3f}\t'
        #    #             f'{new.AIC:.3f}\t'
        #    #             f'{new.AIC-old.AIC:.3f}\t'
        #    #             f'{"Accepted" if result else "Rejected"}')
        #    # return result
        #elif self.criterion == 'BIC':
        #    result = new.bic + self.criterion_threshold < old.bic
        #    logger.info('%s\t%s\tBIC\t%.3f\t%.3f\t%.3f\t%s',
        #                old.model_id,
        #                new.model_id,
        #                old.bic,
        #                new.bic,
        #                new.bic - old.bic,
        #                "Accepted" if result else "Rejected")
        #    # logger.info(f'{old.model_id}\t{new.model_id}\tBIC\t{old.BIC:.3f}\t'
        #    #             f'{new.BIC:.3f}\t'
        #    #             f'{new.BIC-old.BIC:.3f}\t'
        #    #             f'{"Accepted" if result else "Rejected"}')
        #    # return result
        else:
            raise NotImplementedError('Model selection criterion: '
                                      f'{self.criterion}.')
        return result

    def new_model_problem(
            self,
            model: Model,
            criterion: Criterion,
            valid: bool = True,
            autorun: bool = True,
            #model0: Model = None,
            startpoint_latest_mle: bool = True,
    ) -> ModelSelectionProblem:
        """
        Creates a ModelSelectionProblem.

        Arguments
        _________
        model:
            The model description.
        criterion_id:
            The ID of the criterion that should be computed after the model is
            calibrated.
        valid:
            Whether the model should be considered a valid model. If it is not
            valid, it will not be optimized.
        autorun:
            Whether the model should be optimized upon creation.
        #model0:
        #    THe model that the new model `model` was compared to. Used to pass
        #    the maximum likelihood estimate parameters from model
        #    `compared_model_id` to the current model.
        startpoint_latest_mle:
            Whether one start should be initialized at the MLE of the previous
            model `model0`.
        """
        # if compared_model_id in self.selection_history:  FIXME
        #     TODO reconsider, might be a bad idea. also removing parameters
        #     for x_guess that are not estimated in the new model (as is done)
        #     in `row2problem` might also be a bad idea. Both if these would
        #     result in x_guess not actually being the latest MLE.
        #     if compared_model_dict is None:
        #         raise KeyError('For `startpoint_latest_mle`, the information'
        #                        ' of the model that corresponds to the MLE '
        #                        'must be provided. This is to ensure only '
        #                        'estimated parameter values are used in the '
        #                        'startpoint, and all other values are taken '
        #                        'from the PEtab parameter table or the model '
        #                        'specification file.')
        x_guess = None
        if (
            startpoint_latest_mle and
            #model.predecessor_model_id is not None and
            #model.predecessor_model_id != VIRTUAL_INITIAL_MODEL and
            model.predecessor_model_id in self.selection_history
        ):
            #x_guess = \
            #    self.selection_history[model.predecessor_model_id]['MLE']
            predecessor_model = \
                self.selection_history[model.predecessor_model_id]['model']
            x_guess = {
                **predecessor_model.parameters,
                **predecessor_model.estimated_parameters,
            }

        return ModelSelectionProblem(
            model=model,
            criterion=criterion,
            valid=valid,
            autorun=autorun,
            x_guess=x_guess,
            minimize_options=self.minimize_options,
            objective_customizer=self.objective_customizer,
            postprocessor=self.model_postprocessor,
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
