"""Miscellaneous methods."""

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import petab.v1 as petab
import petab_select.ui
from petab.v1.C import ESTIMATE, NOMINAL_VALUE
from petab_select import Model, ModelHash, parameter_string_to_value
from petab_select.constants import PETAB_PROBLEM

from ..history import Hdf5History
from ..objective import Objective
from ..optimize import Optimizer
from ..optimize.ess import (
    SacessOptimizer,
    get_default_ess_options,
)
from ..petab import PetabImporter
from ..problem import Problem

logger = logging.getLogger(__name__)


def model_to_pypesto_problem(
    model: Model,
    objective: Objective = None,
    x_guesses: Iterable[dict[str, float]] = None,
    hierarchical: bool = False,
) -> Problem:
    """Create a pyPESTO problem from a PEtab Select model.

    Parameters
    ----------
    model:
        The model.
    objective:
        The pyPESTO objective.
    x_guesses:
        Startpoints to be used in the multi-start optimization. For example,
        this could be the maximum likelihood estimate from another model.
        Each dictionary has parameter IDs as keys, and parameter values as
        values.
        Values in `x_guess` for parameters that are not estimated will be
        ignored and replaced with their value from the PEtab Select model, if
        defined, else their nominal value in the PEtab parameters table.
    hierarchical:
        Whether the problem involves hierarchical optimization.

    Returns
    -------
    The pyPESTO select problem.
    """
    petab_problem = petab_select.ui.model_to_petab(model=model)[PETAB_PROBLEM]

    corrected_x_guesses = None
    if x_guesses is not None:
        corrected_x_guesses = correct_x_guesses(
            x_guesses=x_guesses,
            model=model,
            petab_problem=petab_problem,
            hierarchical=hierarchical,
        )

    importer = PetabImporter(
        petab_problem,
        hierarchical=hierarchical,
    )
    if objective is None:
        factory = importer.create_objective_creator()
        amici_model = factory.create_model(
            non_estimated_parameters_as_constants=False,
        )
        objective = factory.create_objective(
            model=amici_model,
        )
    pypesto_problem = importer.create_problem(
        objective=objective,
        x_guesses=corrected_x_guesses,
    )
    return pypesto_problem


def model_to_hierarchical_pypesto_problem(*args, **kwargs) -> Problem:
    """Create a hierarchical pyPESTO problem from a PEtab Select model.

    See :func:`model_to_pypesto_problem`.
    """
    pypesto_problem = model_to_pypesto_problem(
        *args,
        **kwargs,
        hierarchical=True,
    )
    return pypesto_problem


def correct_x_guesses(
    x_guesses: Iterable[dict[str, float]],
    model: Model,
    petab_problem: petab.Problem = None,
    hierarchical: bool = False,
):
    """Fix startpoint guesses passed between models of different sizes.

    Any parameter values in `x_guess` for parameters that are not estimated
    should be corrected by replacing them with
    - their corresponding values in `row` if possible, else
    - their corresponding nominal values in the `petab_problem.parameter_df`.

    Parameters
    ----------
    x_guesses:
        The startpoints to correct.
    model:
        The startpoints will be corrected to match this model.
    petab_problem:
        The model's corresponding PEtab problem. If this is not provided,
        it will be created from the `model`.
    hierarchical:
        Whether hierarchical optimization is used.

    Returns
    -------
    The corrected startpoint guesses.
    """
    # TODO reconsider whether correcting is a good idea (`x_guess` is no longer
    # the latest MLE then). Similar todo exists in
    # `ModelSelectorMethod.new_model_problem`.
    # TODO move to PEtab Select?
    # TODO may be issues, e.g. differences in bounds of parameters between
    #      different model PEtab problems is not accounted for, or if there are
    #      different parameters in the old/new PEtab problem.

    if petab_problem is None:
        petab_problem = petab_select.ui.model_to_petab(model=model)[
            PETAB_PROBLEM
        ]

    corrected_x_guesses = None
    if x_guesses is not None:
        corrected_x_guesses = []
        for x_guess in x_guesses:
            corrected_x_guess = []
            for parameter_id in petab_problem.parameter_df.index:
                if hierarchical:
                    if not pd.isna(
                        petab_problem.parameter_df.loc[
                            parameter_id, "parameterType"
                        ]
                    ):
                        continue

                # Use the `x_guess` value, if the parameter is to be estimated.
                if (
                    petab_problem.parameter_df[ESTIMATE].loc[parameter_id] == 1
                    and parameter_id in x_guess
                ):
                    corrected_value = x_guess[parameter_id]
                # Else use the PEtab Select model parameter value, if defined.
                elif parameter_id in model.parameters:
                    corrected_value = parameter_string_to_value(
                        model.parameters[parameter_id]
                    )
                # Default to the PEtab parameter table nominal value
                else:
                    corrected_value = petab_problem.parameter_df.loc[
                        parameter_id, NOMINAL_VALUE
                    ]
                corrected_x_guess.append(corrected_value)
            corrected_x_guesses.append(corrected_x_guess)
    return corrected_x_guesses


class SacessMinimizeMethod:
    """Create a minimize method for SaCeSS that adapts to each problem.

    When a pyPESTO SaCeSS optimizer is created, it takes the problem
    dimension as input. Hence, an optimizer needs to be constructed for
    each problem. Objects of this class act like a minimize method for model
    selection, but a new problem-specific SaCeSS optimizer will be created
    every time a model is minimized.

    Instance attributes correspond to pyPESTO's SaCeSS optimizer, and are
    documented there. Extra keyword arguments supplied to the constructor
    will be passed on to the constructor of the SaCeSS optimizer, for example,
    `max_walltime_s` can be specified in this way. If specified, `tmpdir` will
    be treated as a parent directory.
    """

    def __init__(
        self,
        num_workers: int,
        local_optimizer: Optimizer = None,
        tmpdir: str | Path | None = None,
        save_history: bool = False,
        **optimizer_kwargs,
    ):
        """Construct a minimize-like object."""
        self.num_workers = num_workers
        self.local_optimizer = local_optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.save_history = save_history

        self.tmpdir = tmpdir
        if self.tmpdir is not None:
            self.tmpdir = Path(self.tmpdir)

        if self.save_history and self.tmpdir is None:
            self.tmpdir = Path.cwd() / "sacess_tmpdir"

    def __call__(
        self, problem: Problem, model_hash: ModelHash, **minimize_options
    ):
        """Create then run a problem-specific sacess optimizer.

        Parameters
        ----------
        problem:
            The pyPESTO problem for the model.
        model_hash:
            The model hash.
        minimize_options:
            Passed to :meth:`SacessOptimizer.minimize`.

        Returns
        -------
        The output from :meth:`SacessOptimizer.minimize`.
        """
        # create optimizer
        ess_init_args = get_default_ess_options(
            num_workers=self.num_workers,
            dim=problem.dim,
        )
        for x in ess_init_args:
            x["local_optimizer"] = self.local_optimizer
        model_tmpdir = None
        if self.tmpdir is not None:
            model_tmpdir = self.tmpdir / str(model_hash)
            model_tmpdir.mkdir(exist_ok=False, parents=True)

        ess = SacessOptimizer(
            ess_init_args=ess_init_args,
            tmpdir=model_tmpdir,
            **self.optimizer_kwargs,
        )

        # optimize
        result = ess.minimize(
            problem=problem,
            **minimize_options,
        )

        if self.save_history:
            history_dir = model_tmpdir / "history"
            history_dir.mkdir(exist_ok=False, parents=True)
            for history_index, history in enumerate(ess.histories):
                Hdf5History.from_history(
                    other=history,
                    file=history_dir / (str(history_index) + ".hdf5"),
                    id_=history_index,
                )
        return result
