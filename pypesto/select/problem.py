from typing import Callable, Dict, List, Optional, Set

import petab
from petab.C import ESTIMATE
import petab_select
from petab_select import (
    # FIXME remove calculations here, only perform in PEtab Select
    #calculate_aic,
    #calculate_bic,
    #calculate_aicc,
    Model,
    Criterion,

    #AIC,
    #AICC,
    #BIC,

    #NLLH,
)

from ..objective import ObjectiveBase
from ..optimize import minimize, OptimizerResult
from ..result import Result

from .misc import model_to_pypesto_problem
from .postprocessors import TYPE_POSTPROCESSOR


OBJECTIVE_CUSTOMIZER_TYPE = Callable[[ObjectiveBase], None]
POSTPROCESSOR_TYPE = Callable[["ModelSelectionProblem"], None]


# FIXME rename to ModelProblem? or something else? currently might be confused
#       with `petab_select.Problem`
class ModelSelectionProblem(object):
    """
    Handles the creation, estimation, and evaluation of a model. Here, a model
    is a PEtab problem that is patched with a dictionary of custom parameter
    values (which may specify that the parameter should be estimated).
    Evaluation refers to criterion values such as AIC.
    """
    def __init__(
            self,
            model: Model,
            criterion: Criterion,
            #problem: petab_select.Problem,
            valid: bool = True,
            autorun: bool = True,
            x_guess: List[float] = None,
            x_fixed_estimated: Set[str] = None,
            minimize_options: Dict = None,
            objective_customizer: Optional[OBJECTIVE_CUSTOMIZER_TYPE] = None,
            postprocessor: Optional[TYPE_POSTPROCESSOR] = None,
    ):
        """
        Arguments
        ---------
        model:
            The model description.

        criterion_id:
            The ID of the criterion that should be computed after the model is
            calibrated.

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

        minimize_options:
            Keyword argument options that will be passed on to
            `pypesto.optimize.minimize`.

        objective_customizer:
            A method that takes

        postprocessor:
            A method that takes a `ModelSelectionProblem` as input. For
            example, this can be a function that generates a waterfall plot.
            This postprocessor is applied at the end of the
            `ModelSelectionProblem.set_result` method.

        TODO: constraints
        """
        self.model = model
        self.criterion = criterion
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

        self.model_id = self.model.model_id

        self.postprocessor = postprocessor

        # Criteria
        #self._aic = None
        #self._aicc = None
        #self._bic = None

        if self.valid:
            #self.petab_problem = \
            #    petab.Problem.from_yaml(str(model.petab_yaml))
            # TODO warning/error if x_fixed_estimated is not a parameter ID in
            # the PEtab parameter table. A warning is currently produced in
            # `row2problem` above.
            # Could move to a separate method that is only called when a
            # criterion that requires the number of estimated parameters is
            # called (same for self.n_measurements).
            #self.estimated = x_fixed_estimated | set(
            #    self.petab_problem.parameter_df.query(f'{ESTIMATE} == 1').index
            #)
            #self.n_estimated = len(self.estimated)
            #self.n_measurements = len(self.petab_problem.measurement_df)

            self.pypesto_problem = model_to_pypesto_problem(
                self.model,
                x_guesses=None if x_guess is None else [x_guess],
            )

            if objective_customizer is not None:
                objective_customizer(self.pypesto_problem.objective)

            self.minimize_result = None

            # TODO autorun may be unnecessary now that the `minimize_options`
            # argument is implemented.
            if autorun:
                # If there are no estimated parameters, evaluate the objective
                # function and generate a fake optimization result.
                if not self.pypesto_problem.x_free_indices:
                    self.set_result(
                        create_fake_pypesto_result_from_fval(
                            self.pypesto_problem.objective([])
                        )
                    )
                # TODO rename `minimize_options` to `minimize_kwargs`.
                elif minimize_options:
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
        self.model.set_criterion(Criterion.NLLH, self.optimized_model.fval)
        self.model.compute_criterion(criterion=self.criterion)

        self.model.estimated_parameters = {
            id: value
            for index, (id, value) in enumerate(dict(zip(
                self.pypesto_problem.x_names,
                self.optimized_model.x
            )).items())
            if index in self.pypesto_problem.x_free_indices
        }

        if self.postprocessor is not None:
            self.postprocessor(self)

    #def get_criterion(self, id: str):
    #    """Get a criterion value for the model.

    #    Arguments:
    #        id:
    #            The ID of the criterion (e.g. `'AIC'` or
    #            `petab_select.constants.AIC`).
    #    """
    #    if id == AIC:
    #        return self.aic
    #    elif id == AICC:
    #        return self.aicc
    #    elif id == BIC:
    #        return self.bic
    #    else:
    #        raise NotImplementedError(
    #            f'Unknown criterion: {id}'
    #        )

    #@property
    #def aic(self):
    #    # TODO check naming conflicts, rename to lowercase
    #    if self._aic is None:
    #        self._aic = calculate_aic(
    #            n_estimated=self.n_estimated,
    #            nllh=self.optimized_model.fval,
    #        )
    #        self.model.set_criterion(AIC, self._aic)
    #    return self._aic

    #@property
    #def aicc(self):
    #    # TODO check naming conflicts, rename to lowercase
    #    if self._aicc is None:
    #        # TODO this is probably not how number of priors is meant to be
    #        #      calculated... also untested
    #        n_priors = 0
    #        if self.pypesto_problem.x_priors is not None:
    #            n_priors = len(self.pypesto_problem.x_priors._objectives)
    #        self._aicc = calculate_aicc(
    #            n_estimated=self.n_estimated,
    #            nllh=self.optimized_model.fval,
    #            n_measurements=self.n_measurements,
    #            n_priors=n_priors,
    #        )
    #        self.model.set_criterion(AICC, self._aicc)
    #    return self._aicc

    #@property
    #def bic(self):
    #    if self._bic is None:
    #        # TODO this is probably not how number of priors is meant to be
    #        #      calculated... also untested
    #        n_priors = 0
    #        if self.pypesto_problem.x_priors is not None:
    #            n_priors = len(self.pypesto_problem.x_priors._objectives)
    #        self._bic = calculate_bic(
    #            n_estimated=self.n_estimated,
    #            nllh=self.optimized_model.fval,
    #            n_measurements=self.n_measurements,
    #            n_priors=n_priors,
    #        )
    #        self.model.set_criterion(BIC, self._bic)
    #    return self._bic

    #def compute_all_criteria(self):
    #    self.aic
    #    self.aicc
    #    self.bic


def create_fake_pypesto_result_from_fval(
    fval: float,
) -> Result:
    result = Result()

    optimizer_result = OptimizerResult(
        id='fake_result_for_problem_with_no_estimated_parameters',
        x=[],
        fval=fval,
        grad=None,
        hess=None,
        res=None,
        sres=None,
        n_fval=1,
        n_grad=0,
        n_hess=0,
        n_res=0,
        n_sres=0,
        x0=[],
        fval0=fval,
        history=None,
        exitflag=0,
        time=0.1,
        message='Fake result for problem with no estimated parameters.',
    )

    result.optimize_result.append(optimizer_result)
    return result
