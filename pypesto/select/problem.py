from typing import Callable, Dict, List, Optional, Set

import petab
from petab.C import ESTIMATE
from petab_select import (
    calculate_aic,
    calculate_bic,
    calculate_aicc,
    Model,

    AIC,
    AICC,
    BIC,
)

from ..objective import ObjectiveBase
from ..optimize import minimize
from ..result import Result

from .misc import (
    row2problem,
)

OBJECTIVE_CUSTOMIZER_TYPE = Callable[[ObjectiveBase], None]


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
            valid: bool = True,
            autorun: bool = True,
            x_guess: List[float] = None,
            x_fixed_estimated: Set[str] = None,
            minimize_options: Dict = None,
            objective_customizer: Optional[OBJECTIVE_CUSTOMIZER_TYPE] = None,
    ):
        """
        Arguments
        ---------
        model:
            The model description.

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
        TODO: constraints
        """
        self.model = model
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

        # Criteria
        self._aic = None
        self._aicc = None
        self._bic = None

        if self.valid:
            self.petab_problem = \
                petab.Problem.from_yaml(str(model.petab_yaml))
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
            self.n_measurements = len(self.petab_problem.measurement_df)

            self.pypesto_problem = row2problem(self.model,
                                               self.petab_problem,
                                               x_guess=x_guess)

            if objective_customizer is not None:
                objective_customizer(self.pypesto_problem.objective)

            self.minimize_result = None

            # TODO autorun may be unnecessary now that the `minimize_options`
            # argument is implemented.
            if autorun:
                # TODO rename `minimize_options` to `minimize_kwargs`.
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

        self.model.estimated_parameters = {
            id: value
            for index, (id, value) in enumerate(dict(zip(
                self.pypesto_problem.x_names,
                self.optimized_model.x
            )).items())
            if index in self.pypesto_problem.x_free_indices
        }

    def get_criterion(self, id: str):
        """Get a criterion value for the model.

        Arguments:
            id:
                The ID of the criterion (e.g. `'AIC'` or
                `petab_select.constants.AIC`).
        """
        if id == AIC:
            return self.aic
        elif id == AICC:
            return self.aicc
        elif id == BIC:
            return self.bic
        else:
            raise NotImplementedError(
                f'Unknown criterion: {id}'
            )

    @property
    def aic(self):
        # TODO check naming conflicts, rename to lowercase
        if self._aic is None:
            self._aic = calculate_aic(
                self.n_estimated,
                self.optimized_model.fval,
            )
            self.model.set_criterion(AIC, self._aic)
        return self._aic

    @property
    def aicc(self):
        # TODO check naming conflicts, rename to lowercase
        if self._aicc is None:
            # TODO this is probably not how number of priors is meant to be
            #      calculated... also untested
            n_priors = 0
            if self.pypesto_problem.x_priors is not None:
                n_priors = len(self.pypesto_problem.x_priors._objectives)
            self._aicc = calculate_aicc(
                self.n_estimated,
                self.optimized_model.fval,
                self.n_measurements,
                n_priors,
            )
            self.model.set_criterion(AICC, self._aicc)
        return self._aicc

    @property
    def bic(self):
        if self._bic is None:
            # TODO this is probably not how number of priors is meant to be
            #      calculated... also untested
            n_priors = 0
            if self.pypesto_problem.x_priors is not None:
                n_priors = len(self.pypesto_problem.x_priors._objectives)
            self._bic = calculate_bic(
                self.n_estimated,
                self.optimized_model.fval,
                self.n_measurements,
                n_priors,
            )
            self.model.set_criterion(BIC, self._bic)
        return self._bic

    def compute_all_criteria(self):
        self.aic
        self.aicc
        self.bic
