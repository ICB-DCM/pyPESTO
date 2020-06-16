from typing import Dict, List, Set, Union

import petab
from petab.C import ESTIMATE

from ..optimize import minimize
from ..result import Result

from .constants import MODEL_ID
from .misc import (
    aic,
    bic,
    row2problem,
)


class ModelSelectionProblem(object):
    """
    Handles the creation, estimation, and evaluation of a model. Here, a model
    is a PEtab problem that is patched with a dictionary of custom parameter
    values (which may specify that the parameter should be estimated).
    Evaluation refers to criterion values such as AIC.
    """
    def __init__(
            self,
            row: Dict[str, Union[str, float]],
            petab_problem: petab.problem,
            valid: bool = True,
            autorun: bool = True,
            x_guess: List[float] = None,
            x_fixed_estimated: Set[str] = None,
            minimize_options: Dict = None,
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
        # TODO check naming conflicts, rename to lowercase
        if self._AIC is None:
            self._AIC = aic(self.n_estimated,
                            self.optimized_model.fval)
        return self._AIC

    @property
    def BIC(self):
        if self._BIC is None:
            self._BIC = bic(self.n_estimated,
                            self.n_measurements,
                            self.optimized_model.fval)
        return self._BIC
