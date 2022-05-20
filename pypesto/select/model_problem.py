"""Calibrate a PEtab Select model with pyPESTO."""
import time
from typing import Callable, Dict, List, Optional

from petab_select import Criterion, Model

from ..C import TYPE_POSTPROCESSOR
from ..objective import ObjectiveBase
from ..optimize import minimize
from ..result import OptimizerResult, Result
from .misc import model_to_pypesto_problem

OBJECTIVE_CUSTOMIZER_TYPE = Callable[[ObjectiveBase], None]
POSTPROCESSOR_TYPE = Callable[["ModelProblem"], None]


class ModelProblem:
    """Handles all required calibration tasks on a model.

    Handles the creation, estimation, and evaluation of a model. Here, a model
    is a PEtab problem that is patched with a dictionary of custom parameter
    values (which may specify that the parameter should be estimated).
    Evaluation refers to criterion values such as AIC.

    Attributes
    ----------
    best_start:
        The best start from a pyPESTO optimize result.
    criterion:
        The criterion that should be computed after the model is
        calibrated.
    minimize_options:
        Keyword argument options that will be passed on to
        `pypesto.optimize.minimize`.
    minimize_result:
        A pyPESTO result with an optimize result.
    model:
        A PEtab Select model.
    model_id:
        The ID of the PEtab Select model.
    objective_customizer:
        A method that takes a `pypesto.objective.AmiciObjective` as
        input, and makes changes to the objective in-place.
    postprocessor:
        A method that takes a `ModelSelectionProblem` as input. For
        example, this can be a function that generates a waterfall
        plot. This postprocessor is applied at the end of the
        `ModelProblem.set_result` method.
    pypesto_problem:
        The pyPESTO problem for the model.
    valid:
        If `False`, the model will not be tested.
    x_guess:
        A single startpoint, that will be used as one of the
        startpoints in the multi-start optimization.
    """

    def __init__(
        self,
        model: Model,
        criterion: Criterion,
        valid: bool = True,
        autorun: bool = True,
        x_guess: List[float] = None,
        minimize_options: Dict = None,
        objective_customizer: Optional[OBJECTIVE_CUSTOMIZER_TYPE] = None,
        postprocessor: Optional[TYPE_POSTPROCESSOR] = None,
    ):
        """Construct then calibrate a model problem.

        See the class documentation for documentation of most parameters.

        Parameters
        ----------
        autorun:
            If `False`, the model parameters will not be estimated. Allows
            users to manually call pypesto.minimize with custom options,
            then`set_result()`.

        TODO: constraints
        """
        self.model = model
        self.criterion = criterion
        self.valid = valid

        self.minimize_options = {}
        if minimize_options is not None:
            self.minimize_options = minimize_options

        self.model_id = self.model.model_id
        self.objective_customizer = objective_customizer
        self.postprocessor = postprocessor
        self.best_start = None
        self.minimize_result = None
        self.x_guess = x_guess

        if self.valid:
            self.pypesto_problem = model_to_pypesto_problem(
                self.model,
                x_guesses=None if self.x_guess is None else [self.x_guess],
            )

            if self.objective_customizer is not None:
                self.objective_customizer(self.pypesto_problem.objective)

            # TODO autorun may be unnecessary now that the `minimize_options`
            # argument is implemented.
            if autorun:
                # If there are no estimated parameters, evaluate the objective
                # function and generate a fake optimization result.
                if not self.pypesto_problem.x_free_indices:
                    fake_result_start_time = time.time()
                    fake_result_fval = self.pypesto_problem.objective([])
                    fake_result_evaluation_time = (
                        time.time() - fake_result_start_time
                    )
                    self.set_result(
                        create_fake_pypesto_result_from_fval(
                            fval=fake_result_fval,
                            evaluation_time=fake_result_evaluation_time,
                        )
                    )
                # TODO rename `minimize_options` to `minimize_kwargs`.
                # TODO or allow users to provide custom `minimize` methods?
                elif minimize_options:
                    self.set_result(
                        minimize(self.pypesto_problem, **minimize_options)
                    )
                else:
                    self.set_result(minimize(self.pypesto_problem))

    def set_result(self, result: Result):
        """Postprocess a result.

        Parameters
        ----------
        result:
            A pyPESTO result with an optimize result.
        """
        self.minimize_result = result
        # TODO extract best parameter estimates, to use as start point for
        # subsequent models in model selection, for parameters in those models
        # that were estimated in this model.
        self.best_start = self.minimize_result.optimize_result.list[0]
        self.model.set_criterion(Criterion.NLLH, self.best_start.fval)
        self.model.compute_criterion(criterion=self.criterion)

        self.model.estimated_parameters = {
            id: value
            for index, (id, value) in enumerate(
                dict(
                    zip(
                        self.pypesto_problem.x_names,
                        self.best_start.x,
                    )
                ).items()
            )
            if index in self.pypesto_problem.x_free_indices
        }

        if self.postprocessor is not None:
            self.postprocessor(self)


def create_fake_pypesto_result_from_fval(
    fval: float,
    evaluation_time: float = 0.0,
) -> Result:
    """Create a result for problems with no estimated parameters.

    Parameters
    ----------
    fval:
        The objective function value.
    """
    result = Result()

    optimizer_result = OptimizerResult(
        id="fake_result_for_problem_with_no_estimated_parameters",
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
        time=evaluation_time,
        message="Fake result for problem with no estimated parameters.",
    )

    result.optimize_result.append(optimizer_result)
    return result
